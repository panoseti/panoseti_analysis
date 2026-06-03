# Nextflow Layer C modernization — recipe-driven knobs, run ergonomics, 26.04 hygiene

## Context

The PANOSETI Nextflow pipeline already targets 26.04 and is, on inspection, **substantially
modern**: it uses the native `output {}` publish block (no `publishDir`), `lineage { enabled
= true }`, strict DSL2, and nf-schema 2.5.1 validation wired through the vendored nf-core init
subworkflow. The "suboptimal" feeling comes from three real gaps, not from stale idioms:

1. **Stage parameters are hard-coded into the pipeline, not exposed as config knobs.** The
   calibration science params (`ph_sigma/offset/stride`, `img_stride/block/adc_to_pe`) live in
   **three** divergent places: `nextflow.config` params (exploded into CLI flags by
   `modules/local/calibrate_{ph,img}.nf`), `pa-calibrate` Typer flags, and recipe YAML keys
   consumed only by the pure-Python `pa-run`. The Nextflow path **bypasses** the
   `CalibrationResolver` that CLAUDE.md designates as "the sole source of calibration params",
   so the two orchestrators can produce differently-calibrated L1 from "the same" config. The
   user wants KPF-pipe-style behavior: **stage parameters exposed as knobs in a config/recipe
   file, never hard-coded in pipelines or source.**

2. **Commands are hard to remember.** `nextflow run . -profile laptop --input_obs_dir … --outdir
… --calib_recipe …` is long and unmemorable for newcomers; the pure-Python `pa-run` is far
   easier. There is no task-runner wrapping the canonical invocations.

3. **Half-finished nf-core scaffolding + a resume gap.** `nextflow_schema.json` has drifted from
   `nextflow.config` (7 params missing → `--help` gaps + nf-schema warnings); several template
   files are stale; the publish `mode` is set via an imperative shim rather than the native
   `output {}` block; and `--steps ml` alone produces nothing because `ch_l1_stores` is only
   populated by INGEST (no way to feed pre-existing L1 from disk).

**Intended outcome:** establish the recipe-as-config-knobs pattern (starting with calibration,
via the existing resolver seam), make the pipeline trivially invokable through a `just`
task-runner + cheat-sheet, and close the schema/idiom/resume gaps — all incrementally, each
phase landing green against the existing test suite. **Nextflow stays the production
orchestrator;** none of this adds a second orchestration engine.

### Confirmed decisions (from user)

- **Run UX → Justfile** (`just` recipes wrapping the nextflow invocations) **plus a cheat-sheet doc.**
- **Recipe scope → calibration only** for now: fix the broken calibration recipe schema and route
  the Nextflow calibrate modules through `CalibrationResolver`. This _establishes the pattern_;
  `convert`/`classify` knobs follow the same seam later (see Deferred).
- **Extras included:** nf-core hygiene (schema sync + stale-file scrub), 26.04 idioms (publish
  `mode` into `output{}`, topic-channel version capture), and resume-from-level (`--input_l1_dir`).
- **Not now:** enabling the disabled nf-test gate (the existing `-profile test,laptop` CI smoke
  remains the acceptance gate).

### Key grounding facts (verified in code)

- `recipes/ph_calib_default.yml` / `img_calib_default.yml` use **flat keys** (`ph_sigma`,
  `img_stride`) with **no `calibration:` section** — but `FileCalibrationResolver.resolve()`
  (`io/calibration_source.py:62-66`) reads **nested** `calibration.{ph,img}.{sigma_threshold,…}`.
  So today these recipes would silently resolve to **all pydantic defaults**. They must be
  rewritten before they can drive anything.
- `run_calibrate(recipe=…)` (`adapters/calibrate.py:73-91`) **already** builds the resolver,
  overrides scalars, and stamps `ds.attrs["calibration"]["resolution"]`. The Nextflow modules
  simply never pass `--recipe`. So this is mostly a **Nextflow-side + recipe-file** change.
- When `--recipe` is passed, the resolver fills missing keys with model defaults and
  `run_calibrate` overwrites **every** scalar — so **recipe wins unconditionally**; a scalar flag
  is ignored. The default values must equal the current `nextflow.config` defaults to preserve L1.
- `run_calibrate` does **not** set `ProcessingStep.recipe_name`/`recipe_hash` (`models.py:69-70`),
  unlike `features.py`/`ph_prep.py`. The resolver exposes both (`calibration_source.py:48-54`).
- `conf/test.config` sets `ph_stride=50`/`img_stride=50` via params — load-bearing for the CI
  smoke's statistics on the truncated fixture. This override must migrate to a test recipe.
- `capture_software()` already stamps software versions into every `ProcessingStep`/lineage, so
  topic-channel version capture is an _execution-side_ complement, not new provenance.

---

## Phase 1 — Calibration parameters as recipe knobs (headline)

**Goal:** one recipe file is the single source of calibration knobs, consumed identically by
Nextflow and `pa-run` via `CalibrationResolver`. No calibration param hard-coded in a module
script or `nextflow.config`.

**1a. Rewrite the calibration recipe into the resolver's nested schema.** Replace the two flat
files with a consolidated `recipes/calib_default.yml` (values == current `nextflow.config`
defaults, so L1 array data stays bit-identical):

```yaml
name: calib_default
calibration:
  ph: { sigma_threshold: 5.0, baseline_offset: 800, frame_stride: 200 } # was ph_sigma/ph_offset/ph_stride
  img: { frame_stride: 200, block_size: 8, adc_to_pe: 1.5 } # was img_stride/img_block/img_adc_to_pe
```

Add `recipes/calib_test.yml` (same shape, `frame_stride: 50` for both ph/img) to replace the
`conf/test.config` stride params. Keep/delete the old flat files — delete to avoid confusion.

**1b. Add a `calib_recipe` param** in `nextflow.config` defaulting to the repo recipe (so the
default run is recipe-driven and bit-equivalent):
`calib_recipe = "${projectDir}/recipes/calib_default.yml"`. In `conf/test.config`, set
`calib_recipe = "${projectDir}/recipes/calib_test.yml"` and **remove** the `ph_stride`/`img_stride`
lines. **Remove** the six science params (`ph_sigma/ph_offset/ph_stride/img_stride/img_block/
img_adc_to_pe`) from the `nextflow.config` params block. Add the `calib_recipe` entry to
`nextflow_schema.json` and remove the six dead params' entries (schema sync details in Phase 4).

**1c. Route the calibrate modules through `--recipe` as a staged input** (so recipe content
factors into the Nextflow task hash → `-resume` correctly invalidates on recipe edit). In
`modules/local/calibrate_ph.nf` and `calibrate_img.nf`:

- add `path(recipe)` to `input:`,
- replace `--sigma/--offset/--ph-stride` (resp. `--img-stride/--block/--adc-to-pe`) with
  `--recipe ${recipe}`, keeping the IO flags `--codec/--level/--shard-factor/--lineage-out`.
  In `subworkflows/local/ingest.nf`, pass `file(params.calib_recipe)` into `CALIBRATE_PH`/
  `CALIBRATE_IMG` (a value channel reused across the per-(dp,module) fan-out).

**1d. Stamp recipe provenance on the calibrate step.** In `adapters/calibrate.py`, inside the
existing `if resolution is not None:` block, also set `recipe_name=resolver.recipe_name` and
`recipe_hash=resolver.recipe_hash` on the `ProcessingStep` (`models.py:69-70`). This benefits
`pa-run` identically — no orchestrator-specific branch — and makes "recipe_hash flows through
Nextflow" literally true. **Keep** the scalar kwargs on `run_calibrate` and the Typer flags on
`pa-calibrate` as the **no-recipe fallback** (preserves all existing Python unit tests + ad-hoc
CLI use); only the Nextflow params/flags are removed.

**Critical files:** `recipes/calib_default.yml` (new), `recipes/calib_test.yml` (new),
`modules/local/calibrate_ph.nf`, `modules/local/calibrate_img.nf`,
`subworkflows/local/ingest.nf`, `nextflow.config`, `conf/test.config`,
`src/panoseti_analysis/adapters/calibrate.py`. **Reuse:** `FileCalibrationResolver`,
`CalibrationResolution`, `ds.pano.stamp` — all already shipped.

**Bit-equivalence note:** L1 **array data** stays byte-identical (same resolved params → same
kernel inputs). L1 **attrs differ by design**: the new path adds `calibration.resolution` and
`ProcessingStep.recipe_name/recipe_hash`. Any equivalence check must compare **data_vars**, not
the whole store.

---

## Phase 2 — Run ergonomics: Justfile + cheat-sheet

**Goal:** memorable verbs over the canonical nextflow invocations; `just` for pipeline runs,
the existing `Makefile` stays for dev tasks (`make check`).

**2a. Add a root `Justfile`** (install note: `uv tool install rust-just`, fits the uv workflow).
Profiles = WHERE, recipe = WHAT, sane overridable defaults:

```just
# PANOSETI pipeline runs. Dev tasks (lint/test/check) live in the Makefile.
# Install:  uv tool install rust-just     List:  just --list
profile := "laptop"
outdir  := "results"
recipe  := "recipes/calib_default.yml"

# End-to-end run on the bundled test observation
smoke:
    nextflow run . -profile test,{{profile}} --outdir {{outdir}}

# Ingest a .pffd dir -> L0 + L1.            e.g. just ingest obs.pffd out
ingest input outdir=outdir profile=profile recipe=recipe:
    nextflow run . -profile {{profile}} --steps ingest \
        --input_obs_dir {{input}} --outdir {{outdir}} --calib_recipe {{recipe}}

# Cloud-classify existing L1 stores -> L2.  e.g. just ml results/L1 out   (Phase 3)
ml l1dir outdir=outdir profile=profile:
    nextflow run . -profile {{profile}} --steps ml \
        --input_l1_dir {{l1dir}} --outdir {{outdir}}

# Full ingest,ml pipeline.                  e.g. just run obs.pffd out
run input outdir=outdir profile=profile recipe=recipe:
    nextflow run . -profile {{profile}} --steps ingest,ml \
        --input_obs_dir {{input}} --outdir {{outdir}} --calib_recipe {{recipe}}

# Re-run with -resume (Nextflow cache).     e.g. just resume obs.pffd out
resume input outdir=outdir profile=profile recipe=recipe:
    nextflow run . -profile {{profile}} --steps ingest,ml \
        --input_obs_dir {{input}} --outdir {{outdir}} --calib_recipe {{recipe}} -resume

# Show the resolved parameter schema (--help from nextflow_schema.json)
params:
    nextflow run . --help
```

(Optional dev passthroughs like `check: ; make check` can be added; keep the core focused on runs.)

**2b. Write a concise cheat-sheet** — extend the existing `docs/usage.md` (avoid doc sprawl)
with: the `just` verbs, the profile matrix (laptop/hpc_slurm/ral), the recipe-override pattern,
and the old→new calibration key table from Phase 1a. Point at `just --list` and
`nextflow run . --help` as the self-documenting references. Reference the Justfile from
`README.md` Quick-start and `CLAUDE.md` "How to run".

**Critical files:** `Justfile` (new), `docs/usage.md`, `README.md`, `CLAUDE.md`.

---

## Phase 3 — Resume-from-level (`--input_l1_dir`)

**Goal:** `--steps ml` alone can read existing L1 stores from disk (ingest-then-train-separately,
the RAL flow), instead of producing nothing.

**Mechanics:** add an `input_l1_dir` param (+ schema entry). In `workflows/analysis.nf`, when
`'ml' in steps && !('ingest' in steps)`, build `ch_l1_stores` from disk:
`channel.fromPath("${params.input_l1_dir}/*.L1.zarr", type: 'dir')`, parsing meta from the store
filename (`<run_id>.dp_<dp>.module_<module>.L1.zarr`, the exact name `calibrate_ph.nf` emits) and
deriving `kind` from `dp` via a small ph/img map (img8/img16→img, ph256/ph1024→ph). The `kind`
field is correctness-critical: `ml.nf:15` filters to `kind == 'img'`, so a mis-parsed kind drops
stores. Add a loud guard in `PIPELINE_INITIALISATION` (or `analysis.nf`): exactly one of
`--input_obs_dir`/`--input`/`--input_l1_dir` set for the requested steps.

**Critical files:** `workflows/analysis.nf`, `nextflow.config`, `nextflow_schema.json`,
`subworkflows/local/utils_nfcore_analysis_pipeline/main.nf` (input-source validation).

---

## Phase 4 — nf-core hygiene

**Goal:** `--help` accurate, no nf-schema warnings, no stale template cruft.

- **Sync `nextflow_schema.json` by hand** (do NOT run `nf-core pipelines schema build` — it's
  interactive and would clobber the curated `$defs`/descriptions). Add the 7 drifted params
  (`shard_factor_l0`, `shard_factor_l1`, `ray_launcher`, `slurm_gpu_queue`, `ml_container`,
  `cloud_model_pt`, `cloud_model_json`) plus `calib_recipe` (Phase 1) and `input_l1_dir`
  (Phase 3); remove the six deleted science params. Group logically (e.g. `storage_options`,
  `ml_options`). Verify `--help` shows them and a run emits no "unknown parameter" warnings.
- **Scrub stale files:** `CITATIONS.md` (keep — required by `.nf-core.yml files_exist` — but
  replace nf-core/FASTQ boilerplate with PANOSETI content), `CHANGELOG.md` (PANOSETI v0.1.0
  entry), `tower.yml` (drop the FASTQ wording or delete if Seqera Platform unused), `docs/README.md`
  (strip nf-core/zenodo badges; `is_nfcore: false`), `.nf-core.yml` (prune `files_unchanged`/
  `files_exist` entries pointing at deleted branding/email/igenomes assets), add the standard
  `.editorconfig`.

**Critical files:** `nextflow_schema.json`, `.nf-core.yml`, `CITATIONS.md`, `CHANGELOG.md`,
`tower.yml`, `docs/README.md`, `.editorconfig` (new).

---

## Phase 5 — 26.04 idioms

**Goal:** native publish-mode + execution-side version aggregation.

- **Move publish `mode` into the native `output {}` block.** `outputDir = params.outdir`
  (`nextflow.config:62`) is **not** legacy — it's the current 26.04 base-dir mechanism; keep it
  (fix the misleading "backwards compatibility" comment). **Delete** the imperative
  `workflow.output.mode = params.publish_dir_mode` (`:63`) and instead set `mode params.publish_dir_mode`
  inside the `output {}` block in `main.nf`. Verify the smoke still publishes copies (not symlinks).
- **Topic-channel version capture (lightweight).** Have each `modules/local/*.nf` emit a
  `versions.yml` line (tool + `panoseti_analysis` package version) into a `topic` channel; collect
  via the vendored `utils_nfcore_pipeline` `softwareVersionsToYAML` helper (already referenced by
  the nf-test snapshot) into a published `pipeline_info/pipeline_versions.yml`. This complements
  the data-side `capture_software()` provenance; keep it minimal (one aggregate file, not per-tool
  sprawl).

**Critical files:** `nextflow.config`, `main.nf`, `modules/local/*.nf`,
`subworkflows/local/utils_nfcore_analysis_pipeline/main.nf`.

---

## Deferred / follow-on (not in this plan)

- **Recipe knobs for `convert` + `classify`** (the rest of the KPF generalization): same seam —
  a `convert:`/`classify:` section read by a thin resolver in each adapter; Nextflow passes the
  same recipe file. Calibration here is the template.
- **Enable the disabled nf-test gate** against `-profile test,laptop` (no container), generating
  the `.snap` after Phase 1 (attrs change). User opted out for now.
- **Thread Nextflow native `lid://` lineage ids** into `ProcessingStep.nextflow_lineage_id`.

---

## Verification

Run per phase; full gate at the end. On GPU nodes use `CUDA_VISIBLE_DEVICES=""`.

- **Phase 1:**
  - `make test` green (Option B keeps all calibrate/resolver tests valid).
  - **New unit test:** resolver with `calib_default.yml` reproduces the historical scalar defaults
    (`resolve("ph").params == {sigma_threshold:5.0, baseline_offset:800, frame_stride:200}`, and
    the img analogue) — guards Phase 1a typos.
  - **New equivalence test (acceptance gate):** run `pa-run` and `nextflow run . -profile
test,laptop` on the test `.pffd` with the **same** `calib_recipe`; assert L1 **data_vars**
    identical (`xarray.testing.assert_equal` on data vars only — attrs legitimately differ).
  - CI `pipeline` smoke still produces `L1/*.L1.zarr` + manifests and passes `check_chunk_sizes.py`;
    confirm `calibration.resolution` + `recipe_hash` now appear in L1 attrs/lineage.
- **Phase 2:** `just --list` shows the verbs; `just smoke` reproduces the CI smoke output.
- **Phase 3:** `just ml <prior L1 dir> <out>` (i.e. `--steps ml --input_l1_dir …`) produces L2 +
  L2 manifest; img `kind` preserved through the `ml.nf` filter; the one-input-source guard fires
  when mis-invoked.
- **Phase 4:** `nextflow run . --help` lists all params; a run emits no nf-schema unknown-param
  warnings; stale-file scrub reviewed by diff (nothing the smoke/snapshot depends on is touched).
- **Phase 5:** smoke still publishes **copies**; `pipeline_info/pipeline_versions.yml` written.
- **Final:** `make check` (ruff + ruff format + mypy --strict + pytest) clean, and
  `nextflow run . -profile test,laptop --outdir results_smoke` succeeds with the
  Nextflow-vs-`pa-run` L1 data-var equivalence test green.

## Risks

| Risk                                              | Mitigation                                                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| L1 array regression                               | `calib_default.yml` values must equal current config defaults; the reproduces-defaults unit test + data-var equivalence test pin it. |
| Wrong-schema recipe silently all-defaults         | Phase 1a rewrite into nested `calibration.{ph,img}` is mandatory; reproduces-defaults test catches a missing section.                |
| Breaking `--ph_sigma` CLI overrides               | New override path = edit/point `--calib_recipe`. Scalar flags kept on `pa-calibrate` as no-recipe fallback (Python unaffected).      |
| `conf/test.config` strides go dead                | Migrated to `recipes/calib_test.yml` (stride 50); CI smoke statistics unchanged.                                                     |
| `-resume` reuses stale L1 on recipe edit          | Pass the recipe as a staged `path(recipe)` input so content factors into the task hash.                                              |
| Schema-build clobbers curation                    | Hand-edit `nextflow_schema.json`; never run the interactive tool.                                                                    |
| `input_l1_dir` mis-parsed `kind` drops img stores | Derive `kind` from `dp` via explicit map; guard exactly-one-input-source.                                                            |
