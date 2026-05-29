# panoseti/analysis: Output

Outputs are published **level-major** under `--outdir`. Full format spec:
[`storage_spec.md`](storage_spec.md).

```
<outdir>/
  L0/
    <run>.dp_<p>.module_<m>.zarr/   raw, format-calibrated Zarr (one per (data_product, module))
    hk.<hashset>.zarr/             per-hashset housekeeping telemetry (if hk.pff present)
    manifest.json                  per-run L0 store index + lineage
  L1/
    <run>.dp_<p>.module_<m>.L1.zarr/   science-calibrated Zarr (monotonic unix_t_ns)
    manifest.json                      per-run L1 index + calibration lineage
  pipeline_info/                   Nextflow execution_report/timeline/trace + params.json
```

## L0 — raw view of PFF

One Zarr store per `(data_product, module)` (pypff layout): `images (T,H,W)`,
`unix_t_ns (T,) int64`, header arrays. Native PFF frame order; a `timestamp_qc` record is
attached but timestamps are not reordered.

## L1 — calibrated

- **ph** (`ph256`,`ph1024`): `pedestal_subtracted` (float32) + `hot_pixel_mask`/`dead_pixel_mask`.
- **img** (`img8`,`img16`): `median_subtracted` (float32) + `hot_pixel_mask`/`dead_pixel_mask`.

`unix_t_ns` is guaranteed monotonic-non-decreasing (stable-sorted from L0; frames never
dropped — `pkt_num` carries the original order). Root attrs include `data_level`,
`calibration` (params used), `timestamp_qc`, and the storage version keys.

## manifest.json

Per-run, per-level index with lineage: each store entry carries `dp`, `module`, `level`,
`kind`, `n_frames`, `time_range`, `checksum`, `source_store`, `calibration_params`, and
`timestamp_qc`. See `storage_spec.md` §7.

## Housekeeping

`hk.<hashset>.zarr` stores (per device-type, e.g. `hk.quabo`, `hk.weather`, `hk.mount`) on
a ~1 Hz `hk_t_ns` axis. Absent when the run has no `hk.pff`.

## pipeline_info

Nextflow run reports (`execution_report.html`, `execution_timeline.html`,
`execution_trace.txt`, `pipeline_dag.html`) and the resolved `params.json`.
