# ML dev-velocity refactor — config-driven models, registry, sweeps, dev template

## Context

The spring-cleaning refactor (PR #6) landed and the tight notebook→Ray loop now feels great.
But adding a _new_ model still hits avoidable friction, and one path is outright broken:

- **State loading is broken** — root cause is a hard-coded model-class mismatch, **not**
  `_clone_state` (which is correct). `lab.py::quick_train` trains a `CloudDetectionV2` (the
  default inside `fit_cloud_detector`) but then loads the result into a legacy
  `CloudDetection()` (`ml/cloud-detection/notebooks/lab.py:67`); `io/models.py::load_classifier`
  also defaults to `CloudDetection()`. Both throw a `state_dict` key mismatch for V2 weights.
- **Model name is hard-coded everywhere** — 62 references to `CloudDetection`/`CloudDetectionV2`
  across 12 files; the class is instantiated directly in `cloud_train.py`, `train_cloud.py`,
  `io/models.py`, `lab.py`. Swapping `convnext_tiny`→`convnext_small` or adding a new arch means
  hunting through all of them.
- **A required distributed change is "impossible"** — `fit()` exposes `on_epoch` (end of epoch)
  but no `on_epoch_start`, so the Ray worker cannot call `train_loader.sampler.set_epoch(epoch)`,
  which Ray requires at the **start** of each epoch when `world_size > 1` for shuffling to work
  ([Ray data-loading guide](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html)).
- **Config is scattered** — optimizer/scheduler/hyperparams live across `cloud_train.py`,
  `cloud_v1.yml`, and ad-hoc `hp` dict reads. There is no single typed surface, so an automated
  Ray Tune sweep has nothing clean to vary.

**Goal:** make models config-driven and registered so a new model is "register a class + write a
recipe," fix the state-load bug, unblock multi-GPU training without breaking Layer A purity, add a
Ray Tune sweep adapter, promote the validation plots into a reusable library, and ship a copyable
**`ml/_template/`** batteries-included dev starter. **Non-goal (separate plan):** the high-cadence
imaging-mode pipeline slowness (a convert/calibrate + Zarr-chunking profiling effort).

**Design spine — the promotion gradient.** Three tiers, each looser→tighter:
`ml/<model>/notebooks/lab.py` (volatile R&D, copy `ml/_template/`) → registered package module in
`algorithms/` + a recipe (reliable, recipe-driven) → general shared util in `algorithms/`/`adapters/`.
The "batteries" the template composes ARE the shared utilities built in Workstreams 1–3. Keeps the
existing two-tier ML convention (see `[[ml_two_tier_structure]]`); no divergent top-level `dev/`.

---

## Workstream 1 — Config classes + model/optimizer registries (keystone)

Fixes the state-load bug and de-hardcodes; everything else builds on this.

**New (Layer A, pure torch — passes `tests/algorithms/test_layer_boundary.py`):**

- `algorithms/registry.py` — `@register_model("id")` decorator + `get_model(id) -> type[nn.Module]`
  - `build_model(id, **kw)`. Mirrors the accepted `@qc_check` registry pattern in `algorithms/qc.py`.
- `algorithms/optim.py` — `build_optimizer(params, cfg)` + `build_scheduler(optimizer, cfg)` driven
  by string keys (`optimizer="adamw"`, scheduler params), via small name→class registries.
  Generalizes today's `cloud_train.build_cloud_optimizer` (AdamW + ExponentialLR + ReduceLROnPlateau)
  so changing the optimizer is a one-line recipe edit, not a file hunt.

**New (config — Pydantic contracts, added to `config/models.py` next to `CloudInferParams`):**

- `TrainConfig` (arch id, optimizer name, lr, weight_decay, gamma, scheduler params, batch_size,
  epochs, monitor, minimize). `extra="ignore"` so it can be `model_validate`-d from a larger worker
  config dict; round-trips to/from a flat dict so **Ray Tune `param_space` can override its keys**.

**Wiring (the de-hardcode):**

- Decorate the model classes: `@register_model("cloud_detector")` on `CloudDetection`,
  `@register_model("cloud_detector_v2")` on `CloudDetectionV2` (`algorithms/cloud_detector.py`),
  `@register_model("beta_vae")` on `BetaVAE` (`algorithms/ph_vae.py`).
- `config/models.py::ClassifierBundle` — add `arch: str = "cloud_detector"` (registry id; default
  keeps legacy v1 sidecars loading the legacy CNN). It already carries `model_name` (the run name).
- `io/models.py::load_classifier` — resolve `get_model(bundle.arch)()` instead of the hard-coded
  `CloudDetection()` default. **This is the state-load fix at the I/O boundary.**
- `algorithms/cloud_train.py` (`fit_cloud_detector`, `cloud_val_predictions`),
  `adapters/ray/train_cloud.py`, `ml/.../lab.py::quick_train` — build the model from
  `cfg.arch`/registry, never a literal class. `save_classifier` writes `bundle.arch`.
- `recipes/cloud_v1.yml` & `cloud_v2.yml` — add `arch:` and an `optimizer:` key under `hyperparams`.

**Reuse:** `@qc_check` registry idiom; existing `build_cloud_optimizer` schedule (port verbatim into
`build_optimizer`/`build_scheduler`); `ClassifierBundle.model_name`/`input_spec`; `load_recipe`.

---

## Workstream 2 — Distributed `fit()` hook (+ `_clone_state` hardening)

- `algorithms/training.py::fit` — add `on_epoch_start: Callable[[int], None] | None = None`, invoked
  at the top of the epoch loop **before** iterating `train_loader`. Pure callback; Layer A stays
  framework-free.
- `adapters/ray/train_cloud.py::train_loop_per_worker` — pass
  `on_epoch_start=lambda e: train_loader.sampler.set_epoch(e)` guarded by
  `ray.train.get_context().get_world_size() > 1`. Resolves the "literally impossible" complaint.
- `_clone_state` — return CPU tensors (`v.detach().cpu()`) so checkpoints are device-agnostic
  (standard practice; addresses the user's instinct). The actual load bug is fixed in Workstream 1.

**Reuse:** the existing `fit` loop and the `train_loop_per_worker` DDP wiring (`prepare_model`,
`prepare_data_loader` already present at `train_cloud.py:56,76`).

---

## Workstream 3 — Shared validation-plot library + generic metrics

Promote plotting/metrics off the "model-specific" rung onto "general."

- `algorithms/classification_metrics.py` (Layer A, pure numpy) — move `binary_classification_metrics`,
  `confusion_counts`, `pr_curve_points` out of `cloud_train.py` (they are generic, not cloud-specific).
  Re-export from `cloud_train` for back-compat.
- `adapters/ml/viz.py` (Layer B) — model-agnostic **figure builders** returning matplotlib figures:
  `history_curves`, `confusion_matrix_fig`, `pr_curve_fig`, `histogram_fig`. Single source of truth.
- Refactor `adapters/ml/reporting.py` (W&B logging) and `lab.py::plot_history`/`plot_eval` to call the
  `viz` builders, then `tracker.log_image(...)` / `plt.show()` respectively. No duplicated plotting.

**Reuse:** the existing matplotlib bodies in `reporting.py` and `lab.py` (consolidated, not rewritten).

---

## Workstream 4 — Ray Tune sweep adapter (`pa-tune-cloud`)

Depends on Workstream 1 (the `TrainConfig` dict is the sweep surface).

- `adapters/ml/runner.py` — extract `build_torch_trainer(...)` returning a configured `TorchTrainer`;
  `run_torch_trainer` calls `.fit()` on it (no behavior change); add `run_torch_tuner(...)` that wraps
  the same trainer in `tune.Tuner(trainer, param_space={"train_loop_config": {...}}, tune_config=
TuneConfig(num_samples, metric, mode, scheduler=ASHAScheduler()))` and returns the best trial's
  `(best_state, config, metrics)`.
- `adapters/ray/tune_cloud.py` + `bin/pa-tune-cloud` + `[project.scripts]` entry — reuses the
  **unchanged** `train_loop_per_worker`; reads a new recipe `tune:` block and translates it to a
  search space (`{grid:[...]}→tune.grid_search`, `{uniform:[lo,hi]}→tune.uniform`, `{choice:[...]}
→tune.choice`). Recipes stay "WHAT science" (the sweep space is science config).
- Stamp the winning config + `recipe_hash` into `TrainingProvenance` via the existing `save_classifier`
  path so the best model is fully traceable.

**Reuse:** `train_loop_per_worker`, `report_final_checkpoint`, `init_ray`/launcher modes,
`load_recipe`, `save_classifier`, `TrainingProvenance`. Pattern per
[Ray Tune key concepts](https://docs.ray.io/en/latest/tune/key-concepts.html).

---

## Workstream 5 — `ml/_template/` copyable dev starter + promotion docs

- `ml/_template/` — a copy-me starter: `notebooks/lab.py` (thin; composes registry + `fit` +
  `viz` + `tracking` + `data` with `# your model` / `# your preprocessing` placeholders),
  `notebooks/00_prototype.ipynb` (load features → `quick_train` → `plot_eval` → optional W&B →
  optional `pa-tune-cloud`), `recipes/<name>.yml` (hyperparams + scaling + tune blocks), `README.md`
  documenting the copy → prototype → promote flow.
- The batteries-included helpers move into the shared `adapters/ml/bench.py` (below), so the
  template's `lab.py` is just imports + `# your model` / `# your preprocessing` placeholders; a new
  model needs **no** helper edits — register a class, write a recipe.
- `CLAUDE.md` — short "Promotion gradient" note under the ML/two-tier guidance; update
  `[[ml_two_tier_structure]]` memory.

**Reuse:** existing `ml/cloud-detection/notebooks/lab.py` as the seed for the template.

---

## Shared ML-utils home (anti-drift) — decision

**Yes, consolidate.** Today the R&D helpers (`get_device`, `quick_train`, `plot_history`,
`plot_eval`) are copy-pasted into each `ml/<model>/notebooks/lab.py`, which drifts. The Layer A/B
boundary forbids a single folder spanning pure kernels and I/O, so the shared home is **per layer**:

- **Layer B batteries → `src/panoseti_analysis/adapters/ml/bench.py`** (new): the single importable
  research bench (`get_device`, model-agnostic `quick_train(feature_cache, train_cfg)`, `plot_eval`,
  `plot_history`) built on the registry + `fit` + `viz`. Every `lab.py` and `ml/_template/` imports
  from here instead of redefining; per-model `lab.py` shrinks to genuinely model-specific experiments.
  `adapters/ml/` (data, tracking, runner, reporting, viz, **bench**) thus becomes THE shared ML-utils
  folder — one definition of each helper, imported everywhere.
- **Layer A kernels** are already grouped under `algorithms/` (training, cloud_train, cloud_detector,
  ph_vae, registry, optim, classification_metrics). Optional future tidy: an `algorithms/ml/`
  subpackage to separate ML kernels from data-reduction kernels — **deferred** (high import churn
  across tests, low payoff right now).

This makes "reduce drift" concrete and is the discoverability fix for the "click through many files"
complaint: one obvious place per layer.

---

## Order & verification

Sequence: **W1 → W2 → W3 → W4 → W5** (W1 unblocks all; W2/W3 independent; W4 needs W1; W5 packages 1–3).
Each step lands green against the suite as a trip-wire.

| Change surface                            | Trip-wire test(s)                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------- |
| Layer A purity (registry, optim, metrics) | `tests/algorithms/test_layer_boundary.py`                                                   |
| Registry-driven save/load round-trip      | `tests/io/test_models.py`, new `tests/algorithms/test_registry.py`                          |
| **State-load bug fixed** (acceptance)     | `lab.quick_train` round-trips; new `test_models` V2 case                                    |
| Optimizer/scheduler config                | new `tests/algorithms/test_optim.py`, `tests/algorithms/test_training.py`                   |
| Distributed hook / DDP equivalence        | `tests/adapters/test_ray_equivalence.py`, `test_train_cloud.py` (`CUDA_VISIBLE_DEVICES=""`) |
| Viz/metrics promotion                     | `tests/algorithms/test_cloud_detector.py`, reporting reuse                                  |
| Shared `bench` (anti-drift)               | new `tests/adapters/test_bench.py` (quick_train round-trips on test cache)                  |
| Tune adapter                              | new `tests/adapters/test_tune_cloud.py` (standalone, num_samples=2, 2-lr grid)              |

Per step: targeted module tests → `uv run pytest -q` → `uv run ruff check src tests && uv run mypy
src/panoseti_analysis`. Final: `make check`, plus a `pa-train-cloud` (standalone) train→save→load
round-trip and a `pa-tune-cloud` (standalone) 2-trial smoke on the test feature cache. On GPU nodes
run Ray/equivalence tests with `CUDA_VISIBLE_DEVICES=""` (see `[[gpu_test_caveat]]`).

## Out of scope (separate follow-on plan)

High-cadence imaging-mode pipeline slowness — needs profiling of the convert/calibrate kernels and
Zarr chunk/shard sizing; a distinct subsystem from ML dev-velocity.
