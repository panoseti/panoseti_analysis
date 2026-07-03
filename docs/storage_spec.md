# PANOSETI Analysis Storage Spec

Storage conventions for `panoseti_analysis`. Complements pypff's L0 array spec
([`pypff/docs/zarr_v3_spec.md`](https://github.com/panoseti/pypff/blob/main/docs/zarr_v3_spec.md)), which this builds on.

## §0 Versioning

Two independent version namespaces, both stored as Zarr root attributes:

| Key                                 | Owner     | Governs                                                          |
| ----------------------------------- | --------- | ---------------------------------------------------------------- |
| `panoseti_pff_zarr_version`         | pypff     | L0 array layout (`images`, `unix_t_ns`, header arrays)           |
| `panoseti_analysis_storage_version` | this repo | level structure, manifest schema, HK store, `timestamp_qc`, PACK |

`panoseti_analysis_storage_version = "2.0"` (`MANIFEST_SCHEMA_VERSION = "2.0"` in tandem).
Bumped from 1.0 when `processing_history: list[ProcessingStep]` was added to `StoreLineage`
(breaking change — older manifests without the field still parse, defaulting to `[]`).
`PROVENANCE_SCHEMA_VERSION = "1.0"` governs the `ProcessingStep`/`TrainingProvenance` schema
separately. No backward-compat constraints yet — **bump aggressively**.
Constants live in `src/panoseti_analysis/config/versions.py`.

## §1 Directory layout (level-major)

```
<run>/
  L0/   <run>.dp_<p>.module_<m>.zarr/ …   hk.<hashset>.zarr/   manifest.json   .panoseti-meta/
  L1/   <run>.dp_<p>.module_<m>.zarr/ …                        manifest.json   .panoseti-meta/
  L2/   <run>.cloud.module_<m>.zarr/ …                         manifest.json   .panoseti-meta/
```

- **One store per `(data_product, module)`** — the smallest unit with a coherent shared time
  axis. Never consolidate modules/products (PH is event-based; movie products differ in
  cadence). Cross-module science (SETI multiplicity, gamma stereo) is a **temporal join** over
  the sorted int64 `unix_t_ns` indices via the coincidence-finder kernel — not physical
  consolidation.
- Levels are produced by **publishing** (`output {}` block), not by the converter.

## §2 Per-store arrays

**L0 (from pypff):** required `images (T,H,W)`, `unix_t_ns (T,) int64`; optional header arrays
(single-level `pkt_num`/`pkt_tai`/`pkt_nsec`/`tv_sec`/`tv_usec`/`quabo_num`; module-level
`quabo_<i>_*`). Chunking per pypff (~8 MB pre-compression image chunks).

**L1 (calibrated):**

| Product family        | Required arrays                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------- |
| ph (`ph256`,`ph1024`) | `pedestal_subtracted` float32 `(T,H,W)`; `hot_pixel_mask`,`dead_pixel_mask` uint8 `(H,W)` |
| img (`img8`,`img16`)  | `median_subtracted` float32 `(T,H,W)`; `hot_pixel_mask`,`dead_pixel_mask` uint8 `(H,W)`   |

L1 carries forward all L0 header/timing arrays (incl. `pkt_num`). `unix_t_ns` is
monotonic-non-decreasing (§4). Time-like dims are rechunked uniformly on write. With `shard_factor=0` (default), the
physical zarr chunk is ≤16384 frames. With `shard_factor=N`, each physical shard file
packs N×16384 frames — reducing file counts ~N× on BeeGFS/Expanse without changing read
semantics (all zarr-python ≥3.0 readers handle sharded stores transparently).

**L2 (Derived Products):**

- **Cloud Detector** (`cloud`): One store per `(model, module)`. Time dimension `T_l2`
  (window-centers, monotonic-non-decreasing `int64`). Arrays:

  | Array               | Dtype   | Dims           | Notes                               |
  | ------------------- | ------- | -------------- | ----------------------------------- |
  | `cloud_score`       | float32 | `(T_l2,)`      | P(not-clear); range 0.0–1.0         |
  | `cloud_label`       | uint8   | `(T_l2,)`      | 0=clear, 1=cloudy (≥ threshold)     |
  | `feature_raw_fft`   | float32 | `(T_l2, H, W)` | log-magnitude FFT of stacked frame  |
  | `feature_deriv_fft` | float32 | `(T_l2, H, W)` | log-magnitude FFT of 60 s diff      |
  | `unix_t_ns`         | int64   | `(T_l2,)`      | Window-center timestamps; monotonic |

  Root attrs (in addition to inherited L1 attrs):
  - `data_level = "L2"`
  - `panoseti_analysis_storage_version` (from `config/versions.py`)
  - `inference_params = {"cadence_s": float, "threshold": float}`
  - `model = {"model_name": str, "model_version": str, "checksum": "sha256:<hex>"}`

  L2 manifest lineage entry per store: `StoreLineage` (see `config/models.py`) with fields
  `dp`, `module`, `level="L2"`, `kind="cloud"`, `store`, `n_frames`, `source_store`,
  `model`, `inference_params`.

### Sharding

Sharding (`ShardingCodec`, Zarr v3) packs multiple inner chunks into one physical file,
dramatically reducing inode counts on parallel filesystems.

**Recommended shard factors by platform:**

| Platform       | `shard_factor_l0` | `shard_factor_l1` | L0 files (img16, 13.6M frames)  |
| -------------- | ----------------- | ----------------- | ------------------------------- |
| Laptop / test  | 0                 | 0                 | ~38K                            |
| RAL (BeeGFS)   | 16                | 8                 | ~468                            |
| Expanse (SDSC) | 16                | 8                 | ~468 (required: 2M inode quota) |

Sharding is transparent to readers: `xr.open_zarr`, `zarr.open_group`, and all zarr-python
≥3.0 readers access sharded and unsharded stores identically. Feature caches (produced by
`pa-features-cloud`) use a single monolithic chunk per variable — sharding is not applied
there.

## §3 Root attributes

`data_level` (string validated against the extensible registry in `config/levels.py` — **not**
a hardcoded enum; L0/L1 defined, L2+ open and biased toward future gammapy DL3 compatibility),
plus pypff's `data_product`/`module`/`bytes_per_pixel`/`total_frames`/`frame_config`/
`source_pff_files`/`run_configs`, both `panoseti_*_version` keys, and on L1:
`calibration = {kind, …params}` and `timestamp_qc = {…}` (§4).
All levels carry `processing_history: list[ProcessingStep]` (see `docs/provenance.md`) — an
ordered record of every reduction step that produced this store, stamped by `io/stores.py::write_store`
or `io/stores.py::stamp_history` (for pypff-written L0 stores).

## §4 `timestamp_qc` schema

```json
{
  "status": "clean|repaired|flagged|suspect",
  "monotonic": true,
  "n_frames": 0,
  "n_nonmonotonic": 0,
  "n_duplicates": 0,
  "max_gap_ns": 0,
  "n_gaps_over_threshold": 0,
  "gap_threshold_ns": 0,
  "t_start_ns": 0,
  "t_end_ns": 0
}
```

- **L0** is a faithful native-order PFF mirror; it only _records_ QC (status ∈ clean/flagged/
  suspect — never "repaired").
- **L1+ guarantees monotonic-non-decreasing `unix_t_ns`** — a downstream contract the
  coincidence-finder's binary search depends on. L0→L1 stable-sorts by `unix_t_ns` to repair
  benign non-monotonicity (DAQ real-time buffering), **never drops frames** (duplicates flagged,
  not removed), and sets status `repaired` when reordering occurred, else `flagged`/`clean`.
- **Reversibility:** L1's `pkt_num` array preserves the original acquisition order;
  `np.argsort(L1.pkt_num)` reconstructs the pre-sort frame ordering.
- Gap threshold is data-product-aware: `gap_factor×cadence_ns` for img; `median+10·MAD` of
  inter-arrival intervals for event-based ph.
- `suspect` = a single backward jump exceeds a corruption-grade displacement (img: 1000×
  cadence; ph: 1 s — tunable, calibrate on real data). `pa-calibrate --fail-on-suspect`
  (default true) aborts on `suspect`.

## §5 HK store spec

Per-run, **per-hashset** flat-root `hk.<hashset>.zarr` (e.g. `hk.quabo.zarr`, `hk.weather.zarr`,
`hk.mount.zarr`). `hk.pff` is a single observatory-wide head-node redis-snapshot file, so HK is
**not** per-module (weather/mount aren't module-scoped). Each hashset has a fixed within-run
schema → a clean typed table on its own `hk_t_ns (N,) int64` ~1 Hz axis (no cross-device union).
Quabo hashsets carry `module`/`quabo` as a coordinate. Emitted under `L0/`; raw `hk.pff`
retained in `.panoseti-meta/`.

## §6 Ancillary split

- (a) array-like telemetry → HK stores (§5).
- (b) small scalar configs (daq/obs/network/quabo) → root `run_configs` **and** raw in sidecar.
- (c) logs / manifests / version files (`sw_info`) / sentinels (`run_complete`,
  `recording_ended`) / calibration coeffs (`quabo_ph_baseline`) → `.panoseti-meta/`,
  **propagated to every level** (each level is a self-describing snapshot).

## §7 `manifest.json` (per-run, per-level, with lineage)

```json
{ "panoseti_analysis_storage_version": "2.0", "manifest_schema_version": "2.0",
  "run_id": "obs_…", "level": "L1", "created_utc": "…", "seed_manifest_version": "1.0",
  "stores": [
    { "dp": "ph256", "module": "1", "level": "L1", "kind": "ph",
      "store": "obs_….dp_ph256.module_1.L1.zarr", "n_frames": 16487,
      "time_range": [t0_ns, t1_ns], "checksum": "sha256:…",
      "source_store": "obs_….dp_ph256…zarr",
      "calibration_params": {"kind":"ph","sigma_threshold":5.0,"baseline_offset":800,"frame_stride":200},
      "timestamp_qc": { "...": "..." }, "cadence_ns": null } ] }
```

L0 manifest = pypff store enumeration augmented with `n_frames`/`time_range`/`checksum`. L1
manifest adds `source_store` + `calibration_params` lineage. L2 manifest adds `model` (name, version, checksum)
and `inference_params`. Built by `pa-manifest`: the
adapter does the I/O (reads store attrs, computes checksums); the pure `build_level_manifest`
kernel merges/validates. Manifests are grouped by `run_id` so a multi-run samplesheet yields
one manifest per run per level. Each `StoreLineage` entry echoes the store's `processing_history`; see `docs/provenance.md` for the full spec.

## §8 PACK formats (transfer)

One packed file per `(dp,module,level)` + HK; never Globus raw chunk directories.

- **tar** — compute hop (UCSD→Expanse): fast unpack to Lustre directory stores.
- **Zarr ZipStore** — archive hop (Expanse→Cylon): readable in place
  (`xr.open_zarr("zip://…")` / `zarr.storage.ZipStore`). Uses `ZIP_STORED` (chunks are already
  zstd — no double compression) and ZIP64 (a whole store can exceed 4 GB).
