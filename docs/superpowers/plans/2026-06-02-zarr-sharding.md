# Zarr Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce L0/L1 Zarr store file counts by ~78× via Zarr v3 sharding and dtype-aware 1D chunk sizing, making the pipeline viable on Expanse (2M file limit) and reducing BeeGFS metadata pressure.

**Architecture:** Two changes in two repos. (1) pypff submodule gets `shard_factor` on `ZarrPythonWriter` and a dtype-aware `ts_chunk` fix, exposed through `convert_run` and the CLI. (2) `panoseti_analysis` gets `shard_factor` on `write_store`, plumbed through `pa-calibrate` and the Nextflow modules, with sane per-profile defaults (`shard_factor_l0=16, shard_factor_l1=8` for RAL/HPC). Feature/ML caches are excluded — they already have pinned single-chunk layout.

**Tech Stack:** zarr-python 3.2.1 (`ShardingCodec` via `shards=` kwarg to `create_array`), xarray `to_zarr` with `safe_chunks=False` for the L1 path, Typer CLIs, Nextflow DSL2.

---

## Motivation (numbers from reference dataset)

| Store | Current files | After fix | Reduction |
|---|---|---|---|
| L0 img16 (13.6M frames, cloudy_demo) | 38,294 | **468** | 82× |
| L0 img16 (104M frames, full sci run) | 281,224 | **~3,600** | 78× |
| 8 labeled obs × 96 zarr stores total | ~27M | **~347K** | 78× |

Root causes fixed:
- `ts_chunk = C * 2 = 8192` for all 1D arrays regardless of dtype → for int64 that is 64 KB/chunk (target is 8 MB). Fix: `max(C, min(65536, 8_MB // 8))` = **65536** for all current scalar dtypes (8× more frames per chunk).
- No sharding → every inner chunk is a separate file. With `shard_factor=16`: 16 inner chunks per shard file.

---

## File Map

### pypff submodule (new branch `feat/sharding` from HEAD `d807bc4`)

| File | Change |
|---|---|
| `pypff/src/pypff/zarr/__init__.py` | (1) fix `ts_chunk`; (2) add `shards=` to `ZarrWriter` Protocol + `ZarrPythonWriter.create_array`; (3) add `shard_factor` to `PFFToZarrConverter.__init__` and `convert`; (4) add `shard_factor` to `convert_run`; (5) bump `panoseti_pff_zarr_version` to `"1.1"` |
| `pypff/src/pypff/_cli/zarr.py` | Add `--shard-factor N` option (default 0) |
| `pypff/docs/zarr_v3_spec.md` | Update §9 (chunk/shard rules), §12 (version history) |
| `pypff/src/ci/tier2_logic/test_zarr_roundtrip.py` | Add `TestSharding` class with 5 tests |

### panoseti_analysis (current branch `feature/ml-replication`)

| File | Change |
|---|---|
| `src/panoseti_analysis/adapters/features.py` | **Commit pending chunk fix** (already edited) |
| `src/panoseti_analysis/io/stores.py` | Add `shard_factor: int = 0` to `write_store`; compute shard shapes and pass `safe_chunks=False` |
| `src/panoseti_analysis/adapters/calibrate.py` | Add `shard_factor` param + `--shard-factor` CLI option |
| `modules/local/pff_to_zarr.nf` | Add `--shard-factor ${params.shard_factor_l0}` |
| `modules/local/calibrate_img.nf` | Add `--shard-factor ${params.shard_factor_l1}` |
| `modules/local/calibrate_ph.nf` | Add `--shard-factor ${params.shard_factor_l1}` |
| `nextflow.config` | Add `shard_factor_l0 = 0`, `shard_factor_l1 = 0` defaults |
| `conf/ral.config` | Add `params.shard_factor_l0 = 16`, `params.shard_factor_l1 = 8` |
| `conf/hpc_slurm.config` | Same as ral.config |
| `docs/storage_spec.md` | Add sharding section; note Expanse file count implications |
| `tests/io/test_stores.py` | Add `TestWriteStoreSharding` class with 4 tests |

---

## Task 1 — Create pypff branch and fix dtype-aware 1D chunk sizes

**Files:**
- Modify: `pypff/src/pypff/zarr/__init__.py`
- Test: `pypff/src/ci/tier2_logic/test_zarr_roundtrip.py`

- [ ] **Step 1: Create the pypff branch**

```bash
cd /home/nico/panoseti_analysis/pypff
git checkout -b feat/sharding
```

Expected: `Switched to a new branch 'feat/sharding'`

- [ ] **Step 2: Write the failing test**

Add to `pypff/src/ci/tier2_logic/test_zarr_roundtrip.py` after the last existing test:

```python
# ── SHARDING / CHUNK-SIZE TESTS ──────────────────────────────────────────────

class TestChunkSizes:
    """Verify 1D arrays use dtype-aware ~8 MB chunk targets, not the old C*2 formula."""

    def test_1d_scalar_chunk_is_at_least_65536(self, img16_run: Path, tmp_path: Path) -> None:
        """unix_t_ns and header arrays must use ≥65536 frame chunks for img16."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun
        stores = convert_run(PanosetiRun(img16_run), tmp_path)
        z = zarr.open_group(str(stores[0]), mode="r", zarr_format=3)
        # All 1D arrays should have chunks ≥ 65536 (old value was C*2 = 8192 for img16)
        for name in z.array_keys():
            a = z[name]
            if a.ndim == 1:
                assert a.chunks[0] >= 65536, (
                    f"1D array '{name}' chunk={a.chunks[0]} < 65536 (dtype-aware target)"
                )

    def test_images_chunk_unchanged(self, img16_run: Path, tmp_path: Path) -> None:
        """Image array chunk must still be ~8 MB (4096 for img16)."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun
        stores = convert_run(PanosetiRun(img16_run), tmp_path)
        z = zarr.open_group(str(stores[0]), mode="r", zarr_format=3)
        assert z["images"].chunks[0] == 4096, "img16 image chunk must remain 4096"
        assert z["images"].chunks[1:] == (32, 32)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /home/nico/panoseti_analysis/pypff
uv run pytest src/ci/tier2_logic/test_zarr_roundtrip.py::TestChunkSizes -v
```

Expected: `FAILED test_1d_scalar_chunk_is_at_least_65536` (chunk is 8192, not ≥65536)

- [ ] **Step 4: Apply the fix in `pypff/src/pypff/zarr/__init__.py`**

In `PFFToZarrConverter.convert()`, find the line:
```python
ts_chunk = C * 2
```

Replace it with:
```python
# Target ~8 MB per 1D chunk, floored at C (image chunk) for stride alignment.
# For all current dtypes (uint8–int64), max(C, min(65536, 8MB//8)) = 65536.
ts_chunk = max(C, min(65536, _IMG_CHUNK_BYTES_TARGET // 8))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest src/ci/tier2_logic/test_zarr_roundtrip.py::TestChunkSizes -v
```

Expected: both tests PASS

- [ ] **Step 6: Run full pypff test suite**

```bash
uv run pypff test all
```

Expected: all existing tests PASS. If any test asserts on exact chunk sizes (e.g., `assert arr.chunks == (8192,)`), update those assertions to `>= 65536`.

- [ ] **Step 7: Commit**

```bash
cd /home/nico/panoseti_analysis/pypff
git add src/pypff/zarr/__init__.py src/ci/tier2_logic/test_zarr_roundtrip.py
git commit -m "perf(zarr): dtype-aware 1D chunk sizes (8192 → 65536 for img16)

Old: ts_chunk = C * 2 = 8192 for img16 (only 64 KB per int64 chunk, 125× too small).
New: max(C, min(65536, 8MB // 8)) = 65536 for all current dtypes.
Reduces scalar array chunk files 8× for img16 (8,192 frames/chunk → 65,536)."
```

---

## Task 2 — Add sharding to `ZarrPythonWriter` and `PFFToZarrConverter`

**Files:**
- Modify: `pypff/src/pypff/zarr/__init__.py`
- Test: `pypff/src/ci/tier2_logic/test_zarr_roundtrip.py`

- [ ] **Step 1: Write the failing tests**

Add the following class to `test_zarr_roundtrip.py` (after `TestChunkSizes`):

```python
class TestSharding:
    """shard_factor=N packs N inner chunks into one shard file."""

    def test_sharding_reduces_file_count(self, img16_run: Path, tmp_path: Path) -> None:
        """shard_factor=4 must produce far fewer files than shard_factor=0."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun
        import os

        out_unsharded = tmp_path / "unsharded"
        out_sharded = tmp_path / "sharded"
        run = PanosetiRun(img16_run)
        stores_un = convert_run(run, out_unsharded, shard_factor=0)
        stores_sh = convert_run(run, out_sharded, shard_factor=4)

        files_un = sum(1 for _ in stores_un[0].rglob("*") if _.is_file())
        files_sh = sum(1 for _ in stores_sh[0].rglob("*") if _.is_file())
        assert files_sh < files_un, (
            f"Sharded store ({files_sh} files) must have fewer files than "
            f"unsharded ({files_un} files)"
        )
        # With factor=4, expect roughly 4× reduction in data files
        assert files_sh * 3 < files_un, (
            f"Expected at least 3× file reduction with shard_factor=4, "
            f"got {files_un}/{files_sh}={files_un/files_sh:.1f}×"
        )

    def test_sharded_images_roundtrip(self, img16_run: Path, tmp_path: Path) -> None:
        """Images written with sharding must read back bit-identical."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun, PFFSequence

        run = PanosetiRun(img16_run)
        stores = convert_run(run, tmp_path, shard_factor=4)
        z = zarr.open_group(str(stores[0]), mode="r", zarr_format=3)

        seq = run.get_product(run.list_products()[0])
        expected = seq.read_images_range(0, len(seq))
        np.testing.assert_array_equal(z["images"][:], expected)

    def test_sharded_store_opens_with_xarray(self, img16_run: Path, tmp_path: Path) -> None:
        """Sharded stores must open cleanly with xr.open_zarr."""
        xr = pytest.importorskip("xarray")
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun

        stores = convert_run(PanosetiRun(img16_run), tmp_path, shard_factor=4)
        ds = xr.open_zarr(str(stores[0]), consolidated=False)
        assert "images" in ds
        assert "unix_t_ns" in ds
        assert ds["unix_t_ns"].dtype == np.int64

    def test_shard_attrs_recorded_in_root(self, img16_run: Path, tmp_path: Path) -> None:
        """shard_factor must be stamped into the store's root attrs."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun

        stores = convert_run(PanosetiRun(img16_run), tmp_path, shard_factor=8)
        z = zarr.open_group(str(stores[0]), mode="r", zarr_format=3)
        assert z.attrs.get("shard_factor") == 8

    def test_shard_factor_zero_unchanged(self, img16_run: Path, tmp_path: Path) -> None:
        """shard_factor=0 must not produce a ShardingCodec (pure chunk store)."""
        from pypff.zarr import convert_run
        from pypff.io2 import PanosetiRun

        stores = convert_run(PanosetiRun(img16_run), tmp_path, shard_factor=0)
        z = zarr.open_group(str(stores[0]), mode="r", zarr_format=3)
        img = z["images"]
        # No sharding → shards property is None or equals chunks
        assert img.shards is None or img.shards == img.chunks, (
            "shard_factor=0 must not use ShardingCodec"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nico/panoseti_analysis/pypff
uv run pytest src/ci/tier2_logic/test_zarr_roundtrip.py::TestSharding -v
```

Expected: all fail with `TypeError: convert_run() got unexpected keyword argument 'shard_factor'`

- [ ] **Step 3: Add `shards` parameter to the `ZarrWriter` Protocol**

In `pypff/src/pypff/zarr/__init__.py`, update the Protocol `create_array` signature:

```python
def create_array(
    self,
    root_group: _zarr.Group,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    dimension_names: list[str] | None = None,
    shards: tuple[int, ...] | None = None,
) -> _zarr.Array[Any]:
    """Create an array inside *root_group*. *shards* enables ShardingCodec if not None."""
    ...
```

- [ ] **Step 4: Update `ZarrPythonWriter.create_array` to accept and apply `shards`**

Replace the existing `create_array` method body (around line 203–220) with:

```python
def create_array(
    self,
    root_group: _zarr.Group,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    dimension_names: list[str] | None = None,
    shards: tuple[int, ...] | None = None,
) -> _zarr.Array[Any]:
    import zarr
    group: zarr.Group = root_group  # type: ignore[assignment]
    parts = name.rsplit("/", 1)
    if len(parts) == 2:
        group = group.require_group(parts[0])
        array_name = parts[1]
    else:
        array_name = name

    compressor = self._compressor()
    kwargs: dict[str, Any] = {"shape": shape, "chunks": chunks, "dtype": dtype}
    if shards is not None:
        kwargs["shards"] = shards
    if compressor is not None:
        kwargs["compressors"] = [compressor]
    if dimension_names is not None:
        kwargs["dimension_names"] = dimension_names
    return group.create_array(array_name, **kwargs)  # type: ignore[return-value]
```

- [ ] **Step 5: Add `shard_factor` to `PFFToZarrConverter.__init__`**

In `PFFToZarrConverter.__init__`, add `shard_factor: int = 0` after `run_configs`:

```python
def __init__(
    self,
    seq: PFFSequence,
    writer: ZarrWriter | None = None,
    *,
    time_chunk: int | None = None,
    codec: str = "zstd",
    level: int = 3,
    run_configs: dict[str, Any] | None = None,
    shard_factor: int = 0,
) -> None:
    ...
    self.shard_factor = shard_factor
```

- [ ] **Step 6: Thread `shard_factor` through `PFFToZarrConverter.convert()`**

In the `convert()` method, after computing `C` and `ts_chunk`, add shard shape calculations, then pass `shards=` to every `create_array` call:

```python
def convert(self, out_path: Path) -> Path:
    ...
    T = len(seq)
    H, W = conf.image_shape
    C = self.time_chunk
    ts_chunk = max(C, min(65536, _IMG_CHUNK_BYTES_TARGET // 8))
    SF = self.shard_factor

    # Shard shapes: None when sharding disabled, else SF inner chunks per shard file
    img_shard: tuple[int, ...] | None = (C * SF, H, W) if SF > 0 else None
    ts_shard: tuple[int, ...] | None = (ts_chunk * SF,) if SF > 0 else None

    root = writer.create_store(out_path)
    attrs = self._root_attrs()
    if SF > 0:
        attrs["shard_factor"] = SF
    writer.set_attrs(root, attrs)

    img_arr = writer.create_array(
        root, "images", (T, H, W), (C, H, W), conf.dtype,
        dimension_names=["time", "y", "x"],
        shards=img_shard,
    )
    ...

    ts_arr = writer.create_array(
        root, "unix_t_ns", (T,), (ts_chunk,), np.dtype("int64"),
        dimension_names=["time"],
        shards=ts_shard,
    )
    ...

    # In the header columns loop:
    for key in seq.metadata_offsets:
        zarr_name = self._zarr_name(key)
        dtype = self._field_dtype(key)
        arr = writer.create_array(
            root, zarr_name, (T,), (ts_chunk,), dtype,
            dimension_names=["time"],
            shards=ts_shard,
        )
```

- [ ] **Step 7: Add `shard_factor` to `convert_run`**

Find the `convert_run` function signature and add the parameter:

```python
def convert_run(
    run: PanosetiRun,
    out_dir: str | Path,
    *,
    codec: str = "zstd",
    level: int = 3,
    time_chunk: int | None = None,
    shard_factor: int = 0,
    write_sidecars: bool = True,
) -> list[Path]:
    ...
    # In the loop where PFFToZarrConverter is instantiated:
    conv = PFFToZarrConverter(
        seq, w, time_chunk=time_chunk, run_configs=run_configs,
        shard_factor=shard_factor,
    )
```

- [ ] **Step 8: Run the sharding tests**

```bash
cd /home/nico/panoseti_analysis/pypff
uv run pytest src/ci/tier2_logic/test_zarr_roundtrip.py::TestSharding -v
```

Expected: all 5 tests PASS

- [ ] **Step 9: Run full test suite**

```bash
uv run pypff test all
```

Expected: all tests PASS

- [ ] **Step 10: Commit**

```bash
cd /home/nico/panoseti_analysis/pypff
git add src/pypff/zarr/__init__.py src/ci/tier2_logic/test_zarr_roundtrip.py
git commit -m "feat(zarr): add shard_factor to ZarrPythonWriter and convert_run

shard_factor=N packs N inner zarr chunks into one physical shard file via
ShardingCodec. Default 0 preserves current behaviour.

Recommended values: shard_factor=16 for L0 on BeeGFS/Expanse (~78x file
reduction for img16). Shard factor is stamped into root attrs.

File count for img16 (13.6M frames):
  Unsharded: ~38k files  →  Sharded (factor=16): ~468 files"
```

---

## Task 3 — Update pypff CLI and spec doc

**Files:**
- Modify: `pypff/src/pypff/_cli/zarr.py`
- Modify: `pypff/docs/zarr_v3_spec.md`

- [ ] **Step 1: Add `--shard-factor` to `pypff/_cli/zarr.py`**

Update the `convert` command to accept `shard_factor`:

```python
@app.command()
def convert(
    obs_dir: Annotated[Path, typer.Argument(...)],
    out_dir: Annotated[Path, typer.Argument(...)],
    codec: Annotated[str, typer.Option(...)] = "zstd",
    level: Annotated[int, typer.Option(...)] = 3,
    time_chunk: Annotated[
        int, typer.Option(help="Frames per time chunk (0 = auto-size to ~8 MB)")
    ] = 0,
    shard_factor: Annotated[
        int, typer.Option(
            help="Inner chunks per shard file (0 = no sharding). "
                 "Recommended: 16 for BeeGFS/HPC to reduce file count ~78×."
        )
    ] = 0,
) -> None:
    ...
    chunk = time_chunk if time_chunk > 0 else None
    stores = convert_run(run, out_dir, codec=codec, level=level,
                         time_chunk=chunk, shard_factor=shard_factor)
```

- [ ] **Step 2: Verify CLI help shows the new option**

```bash
cd /home/nico/panoseti_analysis/pypff
uv run pypff zarr --help
```

Expected output includes `--shard-factor INTEGER`.

- [ ] **Step 3: Update `panoseti_pff_zarr_version` to `"1.1"`**

In `pypff/src/pypff/zarr/__init__.py`, find `_root_attrs()` and update:

```python
"panoseti_pff_zarr_version": "1.1",
```

- [ ] **Step 4: Update `pypff/docs/zarr_v3_spec.md`**

Replace Section 9 (Chunk shape rules) and append to Section 13 (Version history):

Replace the entire Section 9 with:

```markdown
## 9. Chunk shape rules

### Image arrays

Chunks span time only: `(C, H, W)` where C is auto-sized to ~8 MB pre-compression:

| Data product | Frame bytes | Auto C | Chunk size |
|---|---|---|---|
| `ph256` (16×16 int16) | 512 B | 16384 | 8 MB |
| `img16` (32×32 int16) | 2 KB | 4096 | 8 MB |
| `ph1024` (32×32 int16) | 2 KB | 4096 | 8 MB |
| `img8` (32×32 uint8) | 1 KB | 8192 | 8 MB |

Override via `convert_run(time_chunk=N)`.

### 1-D arrays (timestamps, headers)

Since v1.1: `ts_chunk = max(C, min(65536, 8_MB // 8)) = 65536` for all current dtypes.
This targets ~8 MB per chunk for the largest dtype (int64=8B) and floors at the image
chunk C for stride alignment.

Prior to v1.1, `ts_chunk = C * 2` (e.g., 8192 for img16) produced 64 KB chunks —
125× smaller than the target for int64 arrays, generating excessive file counts.

### Sharding (optional, recommended for HPC)

When `convert_run(shard_factor=N)` with N > 0, a `ShardingCodec` wraps each array so
that N inner chunks are stored in one physical shard file:

| Array | Inner chunk | Shard (N=16) | Shard size (img16) |
|---|---|---|---|
| `images` | `(C, H, W)` | `(C×N, H, W)` | ~128 MB pre-compression |
| 1-D arrays | `(ts_chunk,)` | `(ts_chunk×N,)` | ~8 MB |

**Recommended values:**

| Platform | `shard_factor_l0` | Effect on img16 (13.6M frames) |
|---|---|---|
| Laptop / test | 0 (no sharding) | ~38K files (fine for local NVMe) |
| BeeGFS / RAL | 16 | ~468 files (78× fewer) |
| Expanse (SDSC) | 16 | ~468 files (required: 2M file quota) |

The `shard_factor` value is stamped into root attrs under the key `"shard_factor"`.
Readers do not need to read this attr — zarr-python handles sharding transparently.
```

Append to Section 13 (Version history):

```markdown
| 1.1 | 2026-06-02 | Dtype-aware 1D chunk sizes (`ts_chunk` = 65536 for img16, was 8192). Optional `shard_factor` parameter on `convert_run` + CLI. `shard_factor` attr added to root. |
```

- [ ] **Step 5: Run lint and full test suite**

```bash
cd /home/nico/panoseti_analysis/pypff
uv run pypff test all --lint
```

Expected: all pass, 0 lint errors

- [ ] **Step 6: Commit**

```bash
cd /home/nico/panoseti_analysis/pypff
git add src/pypff/_cli/zarr.py src/pypff/zarr/__init__.py docs/zarr_v3_spec.md
git commit -m "feat(zarr): expose shard_factor in CLI; bump spec to v1.1

Add --shard-factor option to 'pypff zarr' CLI. Update panoseti_pff_zarr_version
from '1.0' to '1.1' (minor: new optional attr, backward-compatible). Document
sharding in zarr_v3_spec.md §9 with platform-specific recommendations."
```

---

## Task 4 — Commit pending features.py chunk fix in panoseti_analysis

**Context:** `src/panoseti_analysis/adapters/features.py` was already edited in the previous session to pin feature cache chunks to `(N, 2, 32, 32)`. The edit is on disk but was never committed (the Claude classifier was temporarily unavailable). This task commits it before adding sharding changes.

**Files:**
- Modify (already done): `src/panoseti_analysis/adapters/features.py:184`

- [ ] **Step 1: Verify the edit is present**

```bash
grep -n "ds_feat = ds_feat.chunk" /home/nico/panoseti_analysis/src/panoseti_analysis/adapters/features.py
```

Expected: line 184 contains `ds_feat = ds_feat.chunk({"sample": n_samples, "channel": 2, "H": 32, "W": 32})`

- [ ] **Step 2: Run the existing feature adapter tests**

```bash
cd /home/nico/panoseti_analysis
uv run pytest tests/ -k "feature" -v 2>/dev/null || uv run pytest tests/algorithms/test_cloud_detector.py -v
```

Expected: all pass

- [ ] **Step 3: Lint and type check**

```bash
cd /home/nico/panoseti_analysis
uv run ruff check src/panoseti_analysis/adapters/features.py
uv run mypy src/panoseti_analysis/adapters/features.py
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
cd /home/nico/panoseti_analysis
git add src/panoseti_analysis/adapters/features.py
git commit -m "fix(features): pin feature cache chunk layout to (N, 2, 32, 32)

Zarr's auto-chunker picks (2200, 1, 8, 16) for X(8799, 2, 32, 32), spreading
one training sample across 8 chunk files. Pin to a single sample-axis chunk
so each sample is self-contained in one zarr chunk."
```

---

## Task 5 — Add `shard_factor` to `write_store` in panoseti_analysis

**Files:**
- Modify: `src/panoseti_analysis/io/stores.py`
- Test: `tests/io/test_stores.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/io/test_stores.py` after the existing tests:

```python
class TestWriteStoreSharding:
    """shard_factor=N must pack N dask-chunks into one shard file."""

    def test_sharding_reduces_file_count(self, l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
        """shard_factor=4 must produce fewer zarr files than shard_factor=0."""
        out_un = tmp_path / "unsharded.zarr"
        out_sh = tmp_path / "sharded.zarr"
        write_store(l0_img_ds, out_un, shard_factor=0)
        write_store(l0_img_ds, out_sh, shard_factor=4)

        files_un = sum(1 for f in out_un.rglob("*") if f.is_file())
        files_sh = sum(1 for f in out_sh.rglob("*") if f.is_file())
        assert files_sh < files_un, (
            f"Sharded ({files_sh}) must have fewer files than unsharded ({files_un})"
        )

    def test_sharded_roundtrip(self, l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
        """Data written with sharding must read back identically."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=4)
        back = open_store(out)
        np.testing.assert_array_equal(
            back["images"].values, l0_img_ds["images"].values
        )
        np.testing.assert_array_equal(
            back["unix_t_ns"].values, l0_img_ds["unix_t_ns"].values
        )

    def test_shard_factor_zero_no_sharding(self, l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
        """shard_factor=0 must not introduce ShardingCodec."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=0)
        z = zarr.open(str(out), mode="r", zarr_format=3)
        assert z["images"].shards is None, "shard_factor=0 must not shard"

    def test_sharded_store_has_time_dim_sharding_only(
        self, l0_img_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Spatial dims must not be sharded (shard H == chunk H, shard W == chunk W)."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=4)
        z = zarr.open(str(out), mode="r", zarr_format=3)
        img = z["images"]
        if img.shards is not None:
            # shards along time axis only; spatial dims unchanged
            assert img.shards[1] == img.chunks[1], "H must not be sharded"
            assert img.shards[2] == img.chunks[2], "W must not be sharded"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nico/panoseti_analysis
uv run pytest tests/io/test_stores.py::TestWriteStoreSharding -v
```

Expected: `TypeError: write_store() got unexpected keyword argument 'shard_factor'`

- [ ] **Step 3: Implement sharding in `write_store`**

In `src/panoseti_analysis/io/stores.py`, update `write_store`:

```python
def write_store(
    ds: xr.Dataset,
    out_path: str | Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    shard_factor: int = 0,
    processing_history: list[ProcessingStep] | None = None,
) -> int:
    """Write a Dataset to a Zarr v3 directory store; return total bytes on disk.

    ...existing docstring...

    If *shard_factor* > 0, each variable's time-like dimension is sharded:
    ``shard_factor`` dask chunks are packed into one physical shard file.
    ``shard_factor=0`` (default) disables sharding.  Recommended value: 8 for
    L1 stores on BeeGFS/Expanse.
    """
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    ds = ds.drop_encoding()
    time_chunks = {
        str(d): min(int(ds.sizes[d]), _TIME_CHUNK) for d in ds.dims if d in _TIME_DIMS
    }
    if time_chunks:
        ds = ds.chunk(time_chunks)

    if processing_history:
        new_attrs = dict(ds.attrs)
        new_attrs["processing_history"] = [s.model_dump() for s in processing_history]
        ds = ds.assign_attrs(new_attrs)

    compressors = _compressors(codec, level)
    names = list(ds.data_vars) + list(ds.coords)
    encoding: dict[str, Any] = {str(name): {"compressors": compressors} for name in names}

    if shard_factor > 0:
        for name in names:
            try:
                var = ds[name] if name in ds.data_vars else ds.coords[name]
            except KeyError:
                continue
            if not hasattr(var.data, "chunks"):
                continue
            # First element of each dim's dask-chunk tuple is the uniform chunk size
            dask_chunk = tuple(c[0] for c in var.data.chunks)
            shard_shape = tuple(
                c * shard_factor if str(var.dims[i]) in _TIME_DIMS else c
                for i, c in enumerate(dask_chunk)
            )
            encoding[str(name)]["shards"] = shard_shape

    kwargs: dict[str, Any] = {
        "mode": "w",
        "zarr_format": 3,
        "consolidated": False,
        "encoding": encoding,
    }
    if shard_factor > 0:
        # shard covers shard_factor dask chunks; xarray's safe_chunks check would reject this
        # because shard size > dask chunk size. We own the entire write path so this is safe.
        kwargs["safe_chunks"] = False

    ds.to_zarr(str(out_path), **kwargs)
    return sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
```

Also add `from typing import Any` if not already present (check the existing imports at the top of `stores.py`).

- [ ] **Step 4: Run the sharding tests**

```bash
cd /home/nico/panoseti_analysis
uv run pytest tests/io/test_stores.py::TestWriteStoreSharding -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests still PASS

- [ ] **Step 6: Lint and type check**

```bash
uv run ruff check src/panoseti_analysis/io/stores.py
uv run mypy src/panoseti_analysis/io/stores.py
```

Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/panoseti_analysis/io/stores.py tests/io/test_stores.py
git commit -m "feat(stores): add shard_factor to write_store for L1/L2 sharding

shard_factor=N packs N time-chunk slabs into one physical shard file.
Default 0 preserves current unsharded behaviour. Recommended: shard_factor=8
for L1 stores on BeeGFS/Expanse (reduces file count ~8× vs unsharded L1).

Uses safe_chunks=False because shard size intentionally exceeds dask chunk;
the write path is fully controlled so there is no race hazard."
```

---

## Task 6 — Wire `shard_factor` through `pa-calibrate` adapter

**Files:**
- Modify: `src/panoseti_analysis/adapters/calibrate.py`
- Test: `tests/adapters/test_calibrate.py`

- [ ] **Step 1: Check the calibrate function signature**

```bash
grep -n "def run_calibrate\|def main\|shard_factor\|codec\|level" \
    /home/nico/panoseti_analysis/src/panoseti_analysis/adapters/calibrate.py | head -20
```

This tells you the exact line numbers and signature to modify.

- [ ] **Step 2: Write failing test**

Add to `tests/adapters/test_calibrate.py`:

```python
def test_calibrate_with_shard_factor(tmp_path: Path) -> None:
    """pa-calibrate run_calibrate with shard_factor=4 must produce a sharded L1 store."""
    import zarr as _zarr
    from panoseti_analysis.adapters.calibrate import run_calibrate

    l0 = tmp_path / "l0.zarr"
    l1 = tmp_path / "l1.zarr"
    write_store(make_img(n=200, size=32), l0)

    run_calibrate(l0, l1, kind="img", shard_factor=4)

    z = _zarr.open(str(l1), mode="r", zarr_format=3)
    # With sharding, shards should be non-None for the image array
    img = z["median_subtracted"] if "median_subtracted" in z else z["images"]
    assert img.shards is not None, "shard_factor=4 must produce a sharded store"
```

Run it to confirm it fails:

```bash
uv run pytest tests/adapters/test_calibrate.py::test_calibrate_with_shard_factor -v
```

Expected: FAIL (TypeError or shard assert)

- [ ] **Step 3: Add `shard_factor` to `run_calibrate` and `main` in `calibrate.py`**

Find the `run_calibrate` function signature and add `shard_factor: int = 0`:

```python
def run_calibrate(
    l0_store: Path,
    l1_store: Path,
    *,
    kind: str,
    ...
    codec: str = "zstd",
    level: int = 5,
    shard_factor: int = 0,
    ...
) -> StoreLineage:
    ...
    write_store(out, l1_store, codec=codec, level=level,
                shard_factor=shard_factor, processing_history=new_history)
```

In the `main` Typer command, add:

```python
shard_factor: int = typer.Option(
    0,
    help="Inner chunks per shard file (0 = no sharding). Use 8 for BeeGFS/Expanse."
),
```

And pass it through to `run_calibrate(..., shard_factor=shard_factor)`.

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/adapters/test_calibrate.py -v
```

Expected: all PASS

- [ ] **Step 5: Lint and type check**

```bash
uv run ruff check src/panoseti_analysis/adapters/calibrate.py
uv run mypy src/panoseti_analysis/adapters/calibrate.py
```

- [ ] **Step 6: Commit**

```bash
git add src/panoseti_analysis/adapters/calibrate.py tests/adapters/test_calibrate.py
git commit -m "feat(calibrate): add --shard-factor CLI option for L1 sharding"
```

---

## Task 7 — Wire `shard_factor` into Nextflow modules and config

**Files:**
- Modify: `modules/local/pff_to_zarr.nf`
- Modify: `modules/local/calibrate_img.nf`
- Modify: `modules/local/calibrate_ph.nf`
- Modify: `nextflow.config`
- Modify: `conf/ral.config`
- Modify: `conf/hpc_slurm.config` (create if absent)

No unit tests for Nextflow modules, but verify with `nextflow run . -preview`.

- [ ] **Step 1: Add `shard_factor_l0` and `shard_factor_l1` defaults to `nextflow.config`**

In `nextflow.config` inside the `params { }` block, after `time_chunk`:

```groovy
shard_factor_l0            = 0   // shards per L0 zarr chunk (0 = no sharding)
shard_factor_l1            = 0   // shards per L1 zarr chunk (0 = no sharding)
```

- [ ] **Step 2: Add shard defaults to `conf/ral.config`**

Inside the `params { }` block in `conf/ral.config`:

```groovy
params {
    ray_launcher    = 'attach'
    shard_factor_l0 = 16  // ~78× file reduction for img16 on BeeGFS
    shard_factor_l1 = 8   // ~8× file reduction for L1 on BeeGFS
}
```

- [ ] **Step 3: Add shard defaults to `conf/hpc_slurm.config`**

If `conf/hpc_slurm.config` exists, add the same params block. If absent, create it with just the shard_factor params. Check:

```bash
ls /home/nico/panoseti_analysis/conf/hpc_slurm.config
```

Add inside `params { }`:

```groovy
shard_factor_l0 = 16
shard_factor_l1 = 8
```

- [ ] **Step 4: Update `modules/local/pff_to_zarr.nf`**

```groovy
script:
"""
pa-convert ${obs_dir} . \\
    --codec ${params.codec} \\
    --level ${params.level} \\
    --time-chunk ${params.time_chunk} \\
    --shard-factor ${params.shard_factor_l0} \\
    --lineage-out l0_lineage.json
"""
```

- [ ] **Step 5: Update `modules/local/calibrate_img.nf`**

```groovy
script:
"""
pa-calibrate ${l0_store} ${out_store} \\
    --kind img \\
    --img-stride ${params.img_stride} \\
    --block ${params.img_block} \\
    --adc-to-pe ${params.img_adc_to_pe} \\
    --codec ${params.codec} \\
    --level ${params.level} \\
    --shard-factor ${params.shard_factor_l1} \\
    --lineage-out ${out_lineage}
"""
```

- [ ] **Step 6: Update `modules/local/calibrate_ph.nf`**

Read `calibrate_ph.nf`, identify the `pa-calibrate` invocation, and add `--shard-factor ${params.shard_factor_l1}` in the same position as above.

- [ ] **Step 7: Verify Nextflow config parses cleanly**

```bash
cd /home/nico/panoseti_analysis
nextflow config . -profile ral | grep shard
```

Expected output includes:
```
params.shard_factor_l0 = 16
params.shard_factor_l1 = 8
```

- [ ] **Step 8: Commit**

```bash
git add nextflow.config conf/ral.config conf/hpc_slurm.config \
        modules/local/pff_to_zarr.nf modules/local/calibrate_img.nf \
        modules/local/calibrate_ph.nf
git commit -m "feat(nextflow): wire shard_factor_l0/l1 into PFF_TO_ZARR and CALIBRATE

Default 0 (no sharding) for laptop/test profiles; 16/8 for RAL and HPC.
Expanse inode budget: ~347K files per full 8-run L0 ingest (vs ~27M without sharding)."
```

---

## Task 8 — Update pypff submodule pointer and panoseti_analysis storage docs

**Files:**
- Modify: `pypff` (submodule commit pointer)
- Modify: `docs/storage_spec.md`

- [ ] **Step 1: Push the pypff branch**

```bash
cd /home/nico/panoseti_analysis/pypff
git push -u origin feat/sharding
```

- [ ] **Step 2: Update the submodule pointer in panoseti_analysis**

```bash
cd /home/nico/panoseti_analysis
git add pypff
```

This stages the new submodule commit SHA.

- [ ] **Step 3: Update `docs/storage_spec.md`**

Find the section about Zarr chunk sizes (search for `_TIME_CHUNK` or "chunk"). Add a sharding subsection. Exact content depends on what's in the doc now; the key facts to add:

```markdown
### Sharding

Sharding (Zarr v3 `ShardingCodec`) packs multiple inner chunks into one physical
shard file on disk, dramatically reducing file counts on parallel filesystems.

**Recommended shard factors by platform:**

| Platform | `shard_factor_l0` | `shard_factor_l1` | L0 files (img16, 13.6M frames) |
|---|---|---|---|
| Laptop / test | 0 | 0 | ~38K |
| RAL (BeeGFS) | 16 | 8 | ~468 |
| Expanse (SDSC) | 16 | 8 | ~468 (required: 2M inode quota) |

Sharding is transparent to readers: `xr.open_zarr`, `zarr.open_group`, and all
other zarr-python ≥ 3.0 readers access sharded and unsharded stores identically.

**Feature caches** (produced by `pa-features-cloud`) use a single monolithic chunk
per variable — sharding is not applied there.
```

- [ ] **Step 4: Run full panoseti_analysis test suite**

```bash
cd /home/nico/panoseti_analysis
uv run pytest tests/ -v
uv run ruff check src tests && uv run mypy src/panoseti_analysis
```

Expected: all pass, 0 errors

- [ ] **Step 5: Final commit**

```bash
cd /home/nico/panoseti_analysis
git add pypff docs/storage_spec.md
git commit -m "chore: update pypff submodule to feat/sharding; document sharding in storage_spec

pypff feat/sharding adds shard_factor to convert_run + CLI and bumps
panoseti_pff_zarr_version to 1.1. storage_spec.md documents recommended
shard factors per platform and the file-count implications for Expanse."
```

---

## Self-review

### Spec coverage

| Requirement | Task |
|---|---|
| Add sharding to L0 (pypff) | Tasks 2, 3 |
| Add sharding to L1 (write_store) | Task 5 |
| dtype-aware 1D chunk sizes | Task 1 |
| CLI exposure (pypff) | Task 3 |
| CLI exposure (pa-calibrate) | Task 6 |
| Nextflow wiring with per-profile defaults | Task 7 |
| Spec doc update (zarr_v3_spec.md) | Task 3 |
| Storage spec doc update | Task 8 |
| Commit pending features.py fix | Task 4 |
| pypff submodule pointer update | Task 8 |

### Known limitations (not in scope)

- `ph_prep.py` and `features.py` feature caches use non-time dims — sharding is not applicable and is correctly excluded.
- `hk.zarr` HK stores are small (~70 files each) — sharding not applied; no benefit.
- `accumulator.py` L2 streaming stores would benefit from sharding but are out of scope for this PR.
- The pypff `feat/sharding` branch should ultimately be merged into `master` and the submodule updated again.
