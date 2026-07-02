---
title: Contributing
markdownPlugin: checklist
---

# `panoseti/analysis`: Contributing guidelines

Hi there!
Thanks for taking an interest in improving panoseti/analysis.

This page describes the recommended nf-core way to contribute to both panoseti/analysis and nf-core pipelines in general, including:

- [General contribution guidelines](#general-contribution-guidelines): common procedures or guides across all nf-core pipelines.
- [Pipeline-specific contribution guidelines](#pipeline-specific-contribution-guidelines): procedures or guides specific to the development conventions of panoseti/analysis.

## General contribution guidelines

### One-time dev setup

After cloning, run these once to configure local git hooks:

```bash
uv sync                         # install all deps including dev group
uv run nbstripout --install     # auto-strip Jupyter outputs on git add
```

The `nbstripout` filter is registered in `.gitattributes` for `ml/**/*.ipynb` and
`notebooks/**/*.ipynb`. Without running `--install` once per clone, cell outputs
will not be stripped automatically and large PNG blobs may end up in commits.

### Contribution quick start

To contribute code to any nf-core pipeline:

- [ ] Ensure you have Nextflow, nf-core tools, and nf-test installed. See the [nf-core/tools repository](https://github.com/nf-core/tools) for instructions.
- [ ] Check whether a GitHub [issue](https://github.com/panoseti/analysis/issues) about your idea already exists. If an issue does not exist, create one so that others are aware you are working on it.
- [ ] [Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the [panoseti/analysis repository](https://github.com/panoseti/analysis) to your GitHub account.
- [ ] Create a branch on your forked repository and make your changes following [pipeline conventions](#pipeline-contribution-conventions) (if applicable).
- [ ] To fix major bugs, name your branch `patch` and follow the [patch release](#patch-release) process.
- [ ] Update relevant documentation within the `docs/` folder, use nf-core/tools to update `nextflow_schema.json`, and update `CITATIONS.md`.
- [ ] Run and/or update tests. See [Testing](#testing) for more information.
- [ ] [Lint](#lint-tests) your code with nf-core/tools.
- [ ] Submit a pull request (PR) against the `dev` branch and request a review.

If you are not used to this workflow with Git, see the [GitHub documentation](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests) or [Git resources](https://try.github.io/) for more information.

### Testing

Once you have made your changes, run the pipeline with nf-test to test them locally.
For additional information, use the `--verbose` flag to view the Nextflow console log output.

```bash
nf-test test --tag test --profile +docker --verbose
```

If you have added new functionality, ensure you update the test assertions in the `.nf.test` files in the `tests/` directory.
Update the snapshots with the following command:

```bash
nf-test test --tag test --profile +docker --verbose --update-snapshots
```

When you create a pull request with changes, GitHub Actions will run automatic tests.
Pull requests are typically reviewed when these tests are passing.

Two types of tests are typically run:

#### Lint tests

nf-core has a [set of guidelines](https://nf-co.re/docs/specifications/overview) which all pipelines must follow.
To enforce these, run linting with nf-core/tools:

```bash
nf-core pipelines lint <pipeline_directory>
```

If you encounter failures or warnings, follow the linked documentation printed to screen.
For more information about linting tests, see [nf-core/tools API documentation](https://nf-co.re/docs/nf-core-tools/api_reference/latest/pipeline_lint_tests/actions_awsfulltest).

#### Pipeline tests

Each nf-core pipeline should be set up with a minimal set of test data.
GitHub Actions runs the pipeline on this data to ensure it runs through and exits successfully.
If there are any failures then the automated tests fail.
These tests are run with the latest available version of Nextflow and the minimum required version specified in the pipeline code.

### Patch release

> [!WARNING]
> Only in the unlikely event of a release that contains a critical bug.

- [ ] Create a new branch `patch` on your fork based on `upstream/main` or `upstream/master`.
- [ ] Fix the bug and use nf-core/tools to bump the version to the next semantic version, for example, `1.2.3` → `1.2.4`.
- [ ] Open a Pull Request from `patch` directly to `main`/`master` with the changes.

### Pipeline contribution conventions

nf-core semi-standardises how you write code and other contributions to make the panoseti/analysis code and processing logic more understandable for new contributors and to ensure quality.

#### Add a new pipeline step

To contribute a new step to the pipeline, follow the general nf-core coding procedure.
Please also refer to the [pipeline-specific contribution guidelines](#pipeline-specific-contribution-guidelines):

- [ ] Define the corresponding [input channel](#channel-naming-schemes) into your new process from the expected previous process channel.
- [ ] Install a module with nf-core/tools, or write a local module (see [default processes resource requirements](#default-processes-resource-requirements)), and add it to the target `<workflow>.nf`.
- [ ] Define the output channel if needed. Mix the version output channel into `ch_versions` and relevant files into `ch_multiqc`.
- [ ] Add new or updated parameters to `nextflow.config` with a [default value](#default-parameter-values).
- [ ] Add new or updated parameters and relevant help text to `nextflow_schema.json` with [nf-core/tools](#default-parameter-values).
- [ ] Add validation for relevant parameters to the pipeline utilisation section of `utils_nfcore_\_pipeline/main.nf` subworkflow.
- [ ] Perform local tests to validate that the new code works as expected.
  - [ ] If applicable, add a new test in the `tests` directory.
- [ ] Update `usage.md`, `output.md`, and `citation.md` as appropriate.
- [ ] [Lint](#lint-tests) the code with nf-core/tools.
- [ ] Update any diagrams or pipeline images as necessary.

To update the minimum required Nextflow version, see the [Nextflow version bumping](#nextflow-version-bumping) section below. For more information about pipeline contributions, see [pipeline-specific contribution guidelines](#pipeline-specific-contribution-guidelines).

#### Channel naming schemes

Use the following naming schemes for channels to make the channel flow easier to understand:

- Initial process channel: `ch_output_from_<process>`
- Intermediate and terminal channels: `ch_<previousprocess>_for_<nextprocess>`

#### Default parameter values

Parameters should be initialised and defined with default values within the `params` scope in `nextflow.config`.
They should also be documented in the pipeline JSON schema.

To update `nextflow_schema.json`, run:

```bash
nf-core pipelines schema build
```

The schema builder interface that loads in your browser should automatically update the defaults in the parameter documentation.

#### Default processes resource requirements

If you write a local module, specify a default set of resource requirements for the process.

Sensible defaults for process resource requirements (CPUs, memory, time) should be defined in `conf/base.config`.
Specify these with generic `withLabel:` selectors, so they can be shared across multiple processes and steps of the pipeline.

nf-core provides a set of standard labels that you should follow where possible, as seen in the [nf-core pipeline template](https://github.com/nf-core/tools/blob/main/nf_core/pipeline-template/conf/base.config).
These labels define resource defaults for single-core processes, modules that require a GPU, and different levels of multi-core configurations with increasing memory requirements.

Values assigned within these labels can be dynamically passed to a tool using the the `${task.cpus}` and `${task.memory}` Nextflow variables in the `script:` block of a module (see an example in the [modules repository](https://github.com/nf-core/modules/blob/bd1b6a40f55933d94b8c9ca94ec8c1ea0eaf4b82/modules/nf-core/samtools/bam2fq/main.nf#L30)).

#### Nextflow version bumping

If you use a new feature from core Nextflow, bump the minimum required Nextflow version in the pipeline with:

```bash
nf-core pipelines bump-version --nextflow . <min_nf_version>
```

#### Images and figures guidelines

If you update images or graphics, follow the nf-core [style guidelines](https://nf-co.re/docs/community/brand/workflow-schematics).

## Pipeline specific contribution guidelines

### Editing and Previewing Documentation

The documentation for this pipeline is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. The configuration is managed in `mkdocs.yml` at the root of the repository.

To preview the documentation locally while making edits, simply install the `mkdocs-material` theme and run the built-in development server:

```bash
# Install the MkDocs Material theme using uv
uv pip install mkdocs-material

# Start the live-reloading local server
uv run mkdocs serve
```

This will host the documentation at `http://127.0.0.1:8000/`. The page will automatically refresh as you edit files in the `docs/` directory.

### Documentation Hosting

The documentation is automatically published to GitHub Pages using a GitHub Actions workflow (`.github/workflows/docs.yml`). Every time a change is merged into the `main` or `master` branch, the workflow will build the MkDocs site and push the static HTML files to the `gh-pages` branch, which updates the live website automatically. No manual deployment is necessary!
