# Layer A "Spring Cleaning" — KPF-inspired architecture refactor

## Context

Now that the PANOSETI pipeline runs end-to-end, the goal is to simplify the pure-Python
algorithmic layer (Layer A) and reduce boilerplate, drawing on the KPF-Pipeline's clean
separation of concerns — **while keeping the strengths we already have**. After reading
KPF-Pipeline in full and re-mapping our code against it, the verdict is: **we are already
close in spirit.** What we already match:

- **Machine-enforced Layer A purity** (`tests/algorithms/test_layer_boundary.py`) — KPF only
  aspires to this with prose guardrails; we gate it in CI.
- **Single-source-of-truth training engine** (`algorithms/training.py::fit`) reused identically
  by Ray, notebooks, and streaming — KPF's "shared substrate gets a base, glue doesn't" lesson.
- **Two-tier ML lab** (trusted package vs. volatile `lab.py`), **recipes=WHAT / profiles=HOW**
  with `recipe_hash` provenance, and **streaming reuses Layer A wholesale** (`predict_cloud_score`
  is identical across batch/Ray/Serve).

Three gaps where KPF is ahead, and which this refactor closes:

1. **No typed data contract.** The science payload is a raw, schema-less `xr.Dataset` whose
   identity lives in a stringly-typed `attrs` dict. This is the source of our worst boilerplate
   (duplicated attrs-stamping epilogues, scattered defensive `raise ValueError` checks). **#1 lever.**
2. **No QC-alongside-products layer** (KPF has decorator-registered Diagnostics/QC).
3. **No calibration resolution engine** and **no pure-Python dev orchestrator** for the tight
   notebook loop (Nextflow is currently the only Layer C).

### Locked decisions (from user)

- **Migration: incremental in-place.** Additive changes only; each lands green against the
  existing test suite as a regression trip-wire. No big-bang, no parallel v2 package.
- **Data contract: lightweight, via a registered xarray accessor** (`ds.pano.*`) — NOT a wrapper
  class. Rationale: an `xr.Dataset` is already self-describing (dims/coords/attrs), so a KPF-style
  wrapper would re-implement xarray. The accessor centralizes today's smeared type-logic into one
  chokepoint, which _lowers_ the cost of a future KPF-style upgrade (it becomes a mechanical,
  test-guarded, AI-friendly `ds.pano.validate()` → `dp.validate()` change), not a one-way door.
- **Orchestration: thin pure-Python recipe dev-driver** that reuses the existing `run_*()`
  functions in-memory for ONE run. **Nextflow stays** as the production orchestrator
  (resume/scheduling/SLURM). The driver is the canonical readable statement of step order.
- **Calibration: interface + one file-backed impl only.** A `CalibrationResolver` protocol with a
  single recipe/file-backed source today; resolution recorded as a first-class stamped artifact.
  No DB connectors, no nearest-in-time multi-source logic yet (explicit YAGNI).

Plus supporting workstreams: QC layer, adapter de-boilerplate, and adopting KPF driving
principles into `CLAUDE.md`.

---

## Phase ordering & rationale

The `ds.pano` accessor is the keystone — the calibration resolver stamps through it, the recipe
driver validates through it, and the QC runner writes through it — so it lands first. Adapter
de-boilerplate is independent and low-risk, so it lands early to shrink the surface later phases
touch.

1. **Phase 1 — `ds.pano` accessor + `SchemaSpec` registry** (minimum core; unblocks all others)
2. **Phase 2 — Adapter de-boilerplate** (independent; do early)
3. **Phase 3 — `CalibrationResolver` protocol + file-backed impl** (uses Phase 1 stamp seam)
4. **Phase 4 — Thin recipe dev-driver + `bin/pa-run`** (sequences `run_*`, validates via Phase 1)
5. **Phase 5 — QC-alongside-products layer** (writes via Phase 1 attrs seam)
6. **Phase 6 — CLAUDE.md design-principles section** (documents the realized architecture)

---

## Phase 1 — `ds.pano` accessor + `SchemaSpec` registry ⟵ minimum core

**Goal:** one chokepoint for validation + attrs-stamping, replacing the two calibration epilogues
and the scattered consumer var-presence checks.

**New files**

- `src/panoseti_analysis/config/schema.py` — `SchemaSpec` (frozen dataclass: `level`, `kind`,
  `required_data_vars: dict[name, dims]`, `required_dtypes`, `required_attrs`), a
  `dict[(level,kind), SchemaSpec]` registry with `register_schema` / `get_schema`, and the seed
  registrations co-located with the existing level registry. Pure (numpy/xarray/pydantic only).
- `src/panoseti_analysis/config/accessor.py` — `@xr.register_dataset_accessor("pano")` class.
  Imports only `xarray`, `config.{levels,versions,schema,models}`. No `zarr`, no `io`.

Accessor surface:

```python
@xr.register_dataset_accessor("pano")
class PanoAccessor:
    @property
    def level(self) -> str | None: ...          # ds.attrs[DATA_LEVEL_KEY]
    @property
    def kind(self) -> str: ...                   # infer_kind(ds.attrs["data_product"])
    def validate(self, *, level=None, kind=None) -> xr.Dataset:   # raise precise ValueError; chainable; NO I/O
    def stamp(self, *, data_level, calibration=None, timestamp_qc=None,
              carry_from=None, extra_attrs=None) -> xr.Dataset:    # returns NEW ds; sole writer of level/version/calibration keys
```

**Modified files**

- `config/__init__.py` — add `from . import accessor  # noqa: F401` (registration side effect, so
  `ds.pano` exists for anything importing `panoseti_analysis.config`). Defensive duplicate import
  at the top of `io/stores.py` (Layer B, allowed) guards against a Dataset opened before `config`.
- `algorithms/calibrate_img.py` / `calibrate_ph.py` — replace the trailing `out.attrs = {...}`
  epilogue with `out = ds.pano.stamp(data_level="L1", calibration={"kind": kind, **params.model_dump()}, carry_from=ds)`.
  The masks/`carried` construction stays in the kernel (it's math). The `_fix_ph1024_quabo_order`
  path in `calibrate_ph.py` is untouched.
- `algorithms/cloud_detector.py` — replace the inline `if "median_subtracted" not in ds.data_vars:
raise ValueError(...)` (~line 253) with `ds.pano.validate(level="L1", kind="img")`.

**Reuse, do not rewrite:** `config.levels.validate_level` / `infer_kind` (accessor delegates to
these); `config.versions.*` key constants (accessor becomes their sole writer). The calibration
math is untouched.

**zarr round-trip (critical):** the accessor stamps the _same_ attr keys/values the kernels stamp
today, so `write_store` (which reads `ds.attrs`) sees no change. Keep `calibration`/`timestamp_qc`
as JSON-clean dicts (`model_dump(mode="json")`, exactly as `calibrate.py` already does for
`timestamp_qc`) — never pydantic objects or numpy scalars, which break zarr v3 JSON encoding.

**Verification:** `tests/algorithms/test_layer_boundary.py` (purity still holds);
`test_calibrate_img.py`, `test_calibrate_ph.py`, `tests/adapters/test_calibrate.py` (L1 attrs +
masks byte-identical); `test_cloud_detector.py`/`test_cloud_features.py`/`test_cloud_parity.py`
(var-check swap is behavior-neutral); `tests/io/test_stores.py` (attrs round-trip). Add
`tests/config/test_accessor.py` (validate raises on missing var; stamp is non-mutating and
produces the canonical block).

---

## Phase 2 — Adapter de-boilerplate

**Goal:** collapse the ~30-line provenance block copy-pasted across `convert.py`, `calibrate.py`,
`hk.py`, `nextflow/classify_cloud.py`, `ray/classify_cloud.py`, and merge the two near-identical
classify bodies.

**Modified:** `adapters/_common.py` gains the helpers; the five adapters call them.

```python
# _common.py
def build_provenance_step(ds, *, step_name, params, extra_input_checksums=None
    ) -> tuple[ProcessingStep, list[ProcessingStep], list[str], str, dict[str,str]]:
    """Wrap read_history -> last.output_checksum -> now_utc -> capture_software ->
    ProcessingStep(...) -> append_step. NO output checksum (caller writes then checksums)."""

def classify_cloud_store(l1_store, l2_store, *, model, bundle_dict, params, codec, level,
                         quicklook_out=None) -> StoreLineage:
    """Shared body: open L1, build provenance, predict_cloud_score, stamp L2 via
    ds_l2.pano.stamp(data_level='L2', carry_from=ds_l1), write_store, checksum, StoreLineage,
    optional quicklook."""
```

- `nextflow/classify_cloud.py::run_classify` → thin wrapper passing explicit `l2_store`.
- `ray/classify_cloud.py::process_store` stays `@ray.remote`, derives `l2_store` from attrs (its
  only real difference), calls `classify_cloud_store`, returns `(l2_store, record)`. Import
  `panoseti_analysis.config` at module top so the accessor is registered in the Ray worker.

**Reuse:** `read_history`/`append_step`/`capture_software`/`now_utc` (io/provenance), `checksum_store`,
`write_store`, `generate_cloud_quicklook`. Keep output checksum in the caller (needs written bytes).

**Verification:** `tests/adapters/test_ray_equivalence.py` is THE guard (collapsed CPU/Ray bodies
must stay identical); `test_provenance_threading.py`, `test_convert_hk.py`, `test_calibrate.py`,
`tests/io/test_provenance.py`.

---

## Phase 3 — `CalibrationResolver` protocol + file-backed impl

**Goal:** a seam for heterogeneous calibration sources; today one file/recipe-backed source, with
resolution recorded as a first-class stamped artifact.

**New files**

- `config/calibration.py` (pure) — `CalibrationResolver` Protocol (`resolve(cal_type, context) ->
CalibrationResolution`), the `CalibrationResolution` pydantic result (JSON-clean: `cal_type`,
  `source`, `source_hash`, `relevance`, `params`), and a `cal_type -> convention` dispatch table
  (modeled on KPF `CalibrationAssociation`).
- `io/calibration_source.py` (Layer B) — `FileCalibrationResolver`: loads recipe/file via
  `config.recipes.load_recipe`, hashes it, packages params + provenance. No DB, no nearest-in-time.

**Wiring:** `adapters/calibrate.py::run_calibrate` constructs a `FileCalibrationResolver`, resolves
params into `ImgCalibParams`/`PhCalibParams` (defaults must reproduce current values so L1 stays
bit-identical), and records the `CalibrationResolution` in two places: (1) stamped into attrs via
`ds.pano.stamp(calibration={..., "resolution": resolution.model_dump(mode="json")})`, and (2)
`ProcessingStep.params` + `StoreLineage.calibration_params` (already free-form `dict`).

**Reuse:** `config.recipes.load_recipe`, `ImgCalibParams`/`PhCalibParams`, Phase 1 `ds.pano.stamp`.

**Verification:** extend `tests/adapters/test_calibrate.py` (resolution stamped into attrs +
lineage; `open_zarr(l1).attrs["calibration"]["resolution"]["source_hash"]` survives);
`test_calibrate_img/ph` (L1 bit-equivalence); new `tests/config/test_calibration.py` (stable hash,
correct dispatch).

---

## Phase 4 — Thin recipe dev-driver + `bin/pa-run`

**Goal:** the canonical readable statement of step order for ONE run, in memory, mirroring the
Nextflow laptop DAG. Nextflow stays the production orchestrator.

**New files**

- `src/panoseti_analysis/adapters/recipe_driver.py` — `main(stores, recipe)` driver (a flat
  function, < ~120 lines; NOT a DAG engine/scheduler/resume system — anti-framework guardrail).
- `bin/pa-run` shim + `pa-run` entry in `pyproject.toml [project.scripts]`.

```python
def main(stores: StoreLayout, recipe: dict) -> RunOutputs:
    # 1. run_convert(obs_dir, L0/)              -> list[StoreLineage]
    # 2. run_hk(obs_dir, L0/)                   -> list[StoreLineage]   (parallel branch)
    # 3. per (dp,module) L0 store: infer_kind -> run_calibrate(l0, L1/...) (ph or img)
    # 4. per img L1 store: run_classify(l1, L2/..., model)
    # 5. run_manifest(...) per level L0/L1/L2
    # Level-major output layout identical to main.nf output{} block.
```

**Reuse — the whole point:** import and call the existing Layer-B `run_convert`, `run_hk`,
`run_calibrate`, `run_classify`, `run_manifest` verbatim (they already return typed `StoreLineage`/
`Manifest`); `infer_kind` for ph/img routing and the img-only classify gate. Transcribe the DAG
topology literally from `subworkflows/local/ingest.nf` + `ml.nf` and cite them in a comment.

**Verification:** new `tests/adapters/test_recipe_driver.py` (run on the test `.pffd` fixture into a
tmp dir; assert `L0/`/`L1/`/`L2/` stores + manifests). **End-to-end equivalence (acceptance gate):**
`nextflow run . -profile test,laptop --outdir <nf_out>` vs `pa-run <obs> <pa_out> --recipe ...`,
asserting equivalent L0/L1/L2 store sets and matching `median_subtracted`/`pedestal_subtracted`
arrays (deterministic; cloud inference equivalence already covered by `test_ray_equivalence`).

---

## Phase 5 — QC-alongside-products layer

**Goal:** KPF-style decorator-registered QC; a per-`(level,kind)` runner auto-discovers checks,
computes metrics, writes pass/fail + an aggregate `isgood` into attrs / a QC sidecar.

**New files**

- `algorithms/qc.py` (Layer A pure, no sklearn) — `@qc_check(level, kind, key)` decorator tagging
  functions; `run_qc(ds, *, level, kind) -> QCReport` (pydantic: `metrics`, `checks`, `isgood = all(checks)`).
  Seed checks reuse `algorithms/quicklook.py::summarize`/`frame_rate_hz` (frame-rate sanity,
  hot/dead-pixel fraction) and the `TimestampQC` status already produced in calibrate.
- `io/qc.py` (Layer B) — `write_qc_sidecar(report, path)` (reuse `io/quicklook.py::write_summary_json`)
  and stamping via `ds.pano.stamp(extra_attrs={"qc": report.model_dump(mode="json")})`.

**Wiring:** call `run_qc` in the calibrate/classify adapters right after the kernel produces the
dataset, then write/stamp.

**Verification:** `test_layer_boundary.py` (qc.py stays pure); `test_quicklook.py` (reused
semantics); new `tests/algorithms/test_qc.py` (decorator discovery, `isgood` aggregation, a bad
fixture flips a check); new `tests/io/test_qc.py` (sidecar + stamped attr round-trip).

---

## Phase 6 — CLAUDE.md design principles

Add a short "Design principles" section after the three-layer table (additive, non-contradicting):
no hidden state; **fail loudly** (`ds.pano.validate()` raises precise errors at boundaries); **QC
metrics alongside products**; prefer clarity over cleverness / simplest possible implementation;
**anti-framework guardrail** (`recipe_driver.py` is a flat sequence, NOT a second orchestration
engine — Nextflow remains Layer C); **single chokepoints** (`ds.pano` for validate/stamp,
`_common.build_provenance_step` for provenance, `CalibrationResolver` for calibration sourcing).

---

## Cross-cutting trip-wire map

| Change surface                         | Trip-wire test(s)                                                     |
| -------------------------------------- | --------------------------------------------------------------------- |
| Layer A purity (accessor, qc kernel)   | `tests/algorithms/test_layer_boundary.py`                             |
| Stamping absorbed into `ds.pano.stamp` | `test_calibrate_img/ph.py`, `tests/adapters/test_calibrate.py`        |
| Consumer var-check swap                | `test_cloud_detector/_features/_parity.py`                            |
| zarr attrs round-trip                  | `tests/io/test_stores.py`, `tests/io/test_provenance.py`              |
| Provenance de-boilerplate              | `test_provenance_threading.py`, `test_convert_hk.py`                  |
| Classify CPU/Ray collapse              | `tests/adapters/test_ray_equivalence.py` (key guard)                  |
| End-to-end DAG                         | `nextflow run . -profile test,laptop` + new `pa-run` equivalence test |

Per phase: run targeted module tests, then full `pytest -q`, then (Phases 4+) the Nextflow laptop
run. The Ray-equivalence test may be `slow` — run it explicitly for Phase 2. On GPU nodes, run with
`CUDA_VISIBLE_DEVICES=""` per the known equivalence/Ray test caveat.

## Risk register

1. **Accessor registration timing** — register in `config/__init__.py` + defensively in
   `io/stores.py`; idempotent. Highest-impact (everything depends on `ds.pano`).
2. **Non-JSON attrs break zarr round-trip** — always stamp `model_dump(mode="json")` dicts.
3. **Ray worker missing the accessor** — import `config` at module top in the shared classify body.
4. **L1 bit-equivalence regression** via the resolver — defaults must reproduce current params.
5. **Recipe-driver drift from Nextflow DAG** — the laptop equivalence test is the contract;
   transcribe `ingest.nf`/`ml.nf` literally.
6. **Calibration-resolver scope creep** — interface + one file-backed impl only (YAGNI).
7. **QC pulling sklearn into Layer A** — metrics stay in `algorithms/qc.py`, enforced by the
   layer-boundary test.
8. **`pa-run` mistaken for production** — document dev/notebook-only; Nextflow keeps resume/SLURM.

## End-to-end verification

After each phase: `uv run pytest -q` green, `uv run ruff check src tests && uv run mypy
src/panoseti_analysis` clean. After Phase 4+: `nextflow run . -profile test,laptop --outdir
results_smoke` succeeds AND `pa-run` produces equivalent L0/L1/L2 output. Final acceptance: full
`make check` plus the Nextflow-vs-`pa-run` equivalence test.
