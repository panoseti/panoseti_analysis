# panoseti/analysis: Usage

> Parameter reference is generated from `nextflow_schema.json` (`nextflow run . --help`).

## Introduction

The ingest pipeline converts PANOSETI `.pffd` observation runs into calibrated Zarr v3
stores: `PFF → L0 (per-(dp,module)) → L1 (calibrated)`, with lineage manifests and
housekeeping stores. See [`storage_spec.md`](storage_spec.md) for the output format.

## Input

Two ways to specify input:

### 1. Samplesheet (recommended; scales to many runs)

A CSV with a header row and two columns — one row per observation run:

```csv title="samplesheet.csv"
run_id,obs_dir
obs_2024-07-25T04_34_46Z,/expanse/lustre/scratch/$USER/panoseti/inputs/obs_2024-07-25T04_34_46Z.pffd
obs_2024-07-26T05_10_00Z,/expanse/lustre/scratch/$USER/panoseti/inputs/obs_2024-07-26T05_10_00Z.pffd
```

| Column    | Description                                                                           |
| --------- | ------------------------------------------------------------------------------------- |
| `run_id`  | Unique run identifier (becomes `meta.id`/`meta.run_id`; scopes the per-run manifest). |
| `obs_dir` | Path to an existing `.pffd` observation directory.                                    |

```bash
nextflow run . -profile laptop --input samplesheet.csv --outdir <OUTDIR>
```

### 2. Single-run convenience

Skip the samplesheet for a one-off run; the `.pffd` is auto-wrapped into a single run
(`run_id` derived from the directory name):

```bash
nextflow run . -profile laptop --input_obs_dir /path/to/obs.pffd --outdir <OUTDIR>
```

## Running the pipeline

```bash
# bundled truncated test data (CI smoke):
nextflow run . -profile test,laptop --outdir results_smoke

# choose steps (only `ingest` is implemented; reconstruct/ml are stubs):
nextflow run . -profile laptop --steps ingest --input_obs_dir … --outdir …
```

Use `-resume` to reuse cached tasks. Outputs are published level-major under `--outdir`
(`L0/`, `L1/`, each with stores + `manifest.json`).

## Profiles

| Profile     | Executor | Container              | Use                              |
| ----------- | -------- | ---------------------- | -------------------------------- |
| `laptop`    | local    | none (uses uv `.venv`) | development / smoke tests        |
| `docker`    | local    | Docker                 | reproducible local runs          |
| `hpc_slurm` | SLURM    | Apptainer              | SDSC Expanse / any SLURM cluster |

Combine with `test` for the bundled dataset, e.g. `-profile test,laptop`.

### HPC (SDSC Expanse)

```bash
nextflow run . -profile hpc_slurm \
  --input_obs_dir /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
  --outdir        /expanse/lustre/scratch/$USER/panoseti/results \
  --slurm_account <account> --slurm_queue shared -resume
```

Or use the launcher: `bash scripts/launchers/expanse.sh <obs.pffd> <results_dir>`.
Build the container per [`../containers/README.md`](https://github.com/panoseti/panoseti_analysis/blob/main/containers/README.md) and pin it via
`--container` / `conf/hpc_slurm.config`.

## Key parameters

| Param                                              | Default               | Description                                      |
| -------------------------------------------------- | --------------------- | ------------------------------------------------ |
| `--steps`                                          | `ingest`              | Comma-separated steps (`ingest,reconstruct,ml`). |
| `--codec` / `--level`                              | `zstd` / `5`          | Zarr compression.                                |
| `--ph_sigma` / `--ph_offset` / `--ph_stride`       | `5.0` / `800` / `200` | Pulse-height calibration.                        |
| `--img_stride` / `--img_block` / `--img_adc_to_pe` | `200` / `8` / `1.5`   | Movie-mode calibration.                          |

Full list: `nextflow run . --help`.

## Core Nextflow arguments

> [!NOTE]
> These options are part of Nextflow and use a _single_ hyphen (pipeline parameters use a double-hyphen)

### `-profile`

Use this parameter to choose a configuration profile. Profiles can give configuration presets for different compute environments.

Several generic profiles are bundled with the pipeline which instruct the pipeline to use software packaged using different methods (Docker, Singularity, Podman, Shifter, Charliecloud, Apptainer, Conda) - see below.

> [!IMPORTANT]
> We highly recommend the use of Docker or Singularity containers for full pipeline reproducibility, however when this is not possible, Conda is also supported.

Note that multiple profiles can be loaded, for example: `-profile test,docker` - the order of arguments is important!
They are loaded in sequence, so later profiles can overwrite earlier profiles.

If `-profile` is not specified, the pipeline will run locally and expect all software to be installed and available on the `PATH`. This is _not_ recommended, since it can lead to different results on different machines dependent on the computer environment.

- `test`
  - A profile with a complete configuration for automated testing
  - Includes links to test data so needs no other parameters
- `docker`
  - A generic configuration profile to be used with [Docker](https://docker.com/)
- `singularity`
  - A generic configuration profile to be used with [Singularity](https://sylabs.io/docs/)
- `podman`
  - A generic configuration profile to be used with [Podman](https://podman.io/)
- `shifter`
  - A generic configuration profile to be used with [Shifter](https://nersc.gitlab.io/development/shifter/how-to-use/)
- `charliecloud`
  - A generic configuration profile to be used with [Charliecloud](https://charliecloud.io/)
- `apptainer`
  - A generic configuration profile to be used with [Apptainer](https://apptainer.org/)
- `wave`
  - A generic configuration profile to enable [Wave](https://seqera.io/wave/) containers. Use together with one of the above (requires Nextflow ` 24.03.0-edge` or later).
- `conda`
  - A generic configuration profile to be used with [Conda](https://conda.io/docs/). Please only use Conda as a last resort i.e. when it's not possible to run the pipeline with Docker, Singularity, Podman, Shifter, Charliecloud, or Apptainer.

### `-resume`

Specify this when restarting a pipeline. Nextflow will use cached results from any pipeline steps where the inputs are the same, continuing from where it got to previously. For input to be considered the same, not only the names must be identical but the files' contents as well. For more info about this parameter, see [this blog post](https://www.nextflow.io/blog/2019/demystifying-nextflow-resume.html).

You can also supply a run name to resume a specific run: `-resume [run-name]`. Use the `nextflow log` command to show previous run names.

### `-c`

Specify the path to a specific config file (this is a core Nextflow command). See the [nf-core website documentation](https://nf-co.re/usage/configuration) for more information.

## Custom configuration

### Resource requests

Whilst the default requirements set within the pipeline will hopefully work for most people and with most input data, you may find that you want to customise the compute resources that the pipeline requests. Each step in the pipeline has a default set of requirements for number of CPUs, memory and time. For most of the pipeline steps, if the job exits with any of the error codes specified [here](https://github.com/nf-core/rnaseq/blob/4c27ef5610c87db00c3c5a3eed10b1d161abf575/conf/base.config#L18) it will automatically be resubmitted with higher resources request (2 x original, then 3 x original). If it still fails after the third attempt then the pipeline execution is stopped.

To change the resource requests, please see the [max resources](https://nf-co.re/docs/running/configuration/nextflow-for-your-system#set-max-resources) and [customise process resources](https://nf-co.re/docs/running/configuration/nextflow-for-your-system#customize-process-resources) section of the nf-core website.

### Custom Containers

In some cases, you may wish to change the container or conda environment used by a pipeline steps for a particular tool. By default, nf-core pipelines use containers and software from the [biocontainers](https://biocontainers.pro/) or [bioconda](https://bioconda.github.io/) projects. However, in some cases the pipeline specified version maybe out of date.

To use a different container from the default container or conda environment specified in a pipeline, please see the [updating tool versions](https://nf-co.re/docs/running/configuration/nextflow-for-your-system#update-tool-versions) section of the nf-core website.

### Custom Tool Arguments

A pipeline might not always support every possible argument or option of a particular tool used in pipeline. Fortunately, nf-core pipelines provide some freedom to users to insert additional parameters that the pipeline does not include by default.

## Running in the background

Nextflow handles job submissions and supervises the running jobs. The Nextflow process must run until the pipeline is finished.

The Nextflow `-bg` flag launches Nextflow in the background, detached from your terminal so that the workflow does not stop if you log out of your session. The logs are saved to a file.

Alternatively, you can use `screen` / `tmux` or similar tool to create a detached session which you can log back into at a later time.
Some HPC setups also allow you to run nextflow within a cluster job submitted your job scheduler (from where it submits more jobs).

## Nextflow memory requirements

In some cases, the Nextflow Java virtual machines can start to request a large amount of memory.
We recommend adding the following line to your environment to limit this (typically in `~/.bashrc` or `~./bash_profile`):

```bash
NXF_OPTS='-Xms1g -Xmx4g'
```
