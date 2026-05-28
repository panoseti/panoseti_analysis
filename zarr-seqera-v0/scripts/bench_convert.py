#!/usr/bin/env python3
"""
Benchmark PFF → Zarr conversion across codecs, levels, and chunk sizes.

Usage:
    python scripts/bench_convert.py <obs.pffd> <scratch_dir> [--out results.tsv]

Output: TSV with columns:
    product  codec  level  time_chunk  elapsed_s  read_mb_s  write_mb_s
    compress_ratio  peak_rss_mb  n_frames  zarr_mb  pff_mb
"""
from __future__ import annotations

import resource
import shutil
import sys
import time
import tracemalloc
from itertools import product
from pathlib import Path

import typer

app = typer.Typer()

CODECS = ["zstd", "blosc-lz4", "none"]
LEVELS = [1, 3, 5]
TIME_CHUNKS = [2048, 8192, 16384]


def _dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on Linux, kilobytes on macOS
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1024**2
    return usage.ru_maxrss / 1024


def bench_one(
    run,
    product_name: str,
    out_path: Path,
    codec: str,
    level: int,
    time_chunk: int,
) -> dict:
    from pypff.zarr import PFFToZarrConverter, ZarrPythonWriter

    seq = run.get_product(product_name)
    pff_mb = sum(p.stat().st_size for p in seq.file_paths) / 1024**2

    if out_path.exists():
        shutil.rmtree(out_path)

    writer = ZarrPythonWriter(codec=codec, level=level)
    conv = PFFToZarrConverter(seq, writer, time_chunk=time_chunk)

    tracemalloc.start()
    rss_before = _peak_rss_mb()
    t0 = time.monotonic()

    conv.convert(out_path)

    elapsed = time.monotonic() - t0
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _peak_rss_mb()

    zarr_mb = _dir_mb(out_path)
    compress_ratio = pff_mb / zarr_mb if zarr_mb > 0 else float("inf")

    return {
        "product": product_name,
        "codec": codec,
        "level": level,
        "time_chunk": time_chunk,
        "elapsed_s": round(elapsed, 2),
        "read_mb_s": round(pff_mb / elapsed, 1) if elapsed > 0 else 0,
        "write_mb_s": round(zarr_mb / elapsed, 1) if elapsed > 0 else 0,
        "compress_ratio": round(compress_ratio, 3),
        "peak_rss_mb": round(rss_after - rss_before, 1),
        "peak_traced_kb": round(peak_traced / 1024, 1),
        "n_frames": len(seq),
        "zarr_mb": round(zarr_mb, 2),
        "pff_mb": round(pff_mb, 2),
    }


@app.command()
def main(
    obs_dir: Path = typer.Argument(..., help="Input .pffd observation directory"),
    scratch_dir: Path = typer.Argument(..., help="Temporary directory for Zarr outputs"),
    out: Path = typer.Option(Path("bench_results.tsv"), help="Output TSV file"),
    codecs: str = typer.Option(",".join(CODECS), help="Comma-separated codec list"),
    levels: str = typer.Option(",".join(map(str, LEVELS)), help="Comma-separated level list"),
    chunks: str = typer.Option(",".join(map(str, TIME_CHUNKS)), help="Comma-separated chunk list"),
) -> None:
    """Sweep codec × level × chunk configurations and report throughput + compression."""
    from pypff.io2 import PanosetiRun

    run = PanosetiRun(obs_dir)
    products = run.list_products()
    if not products:
        typer.echo("No data products found.", err=True)
        raise typer.Exit(1)

    _codecs = [c.strip() for c in codecs.split(",")]
    _levels = [int(v) for v in levels.split(",")]
    _chunks = [int(v) for v in chunks.split(",")]

    total_configs = len(products) * len(_codecs) * len(_levels) * len(_chunks)
    typer.echo(f"Benchmarking {len(products)} product(s) × {len(_codecs)} codecs × "
               f"{len(_levels)} levels × {len(_chunks)} chunks = {total_configs} runs")
    typer.echo(f"Output: {out}\n")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for prod_name, codec, level, chunk in product(products, _codecs, _levels, _chunks):
        if codec == "none" and level != _levels[0]:
            continue  # level is irrelevant for no-compression; run once
        label = f"{prod_name}  {codec}  lv={level}  chunk={chunk}"
        typer.echo(f"  {label} … ", nl=False)
        out_path = scratch_dir / f"{prod_name}_{codec}_{level}_{chunk}.zarr"
        try:
            row = bench_one(run, prod_name, out_path, codec, level, chunk)
            typer.echo(
                f"{row['elapsed_s']:.1f}s  "
                f"read={row['read_mb_s']} MB/s  "
                f"write={row['write_mb_s']} MB/s  "
                f"ratio={row['compress_ratio']:.2f}×"
            )
            rows.append(row)
        except Exception as exc:
            typer.echo(f"FAILED: {exc}", err=True)
            rows.append({"product": prod_name, "codec": codec, "level": level,
                         "time_chunk": chunk, "error": str(exc)})

    if not rows:
        typer.echo("No results.", err=True)
        raise typer.Exit(1)

    # Write TSV
    cols = ["product", "codec", "level", "time_chunk", "elapsed_s", "read_mb_s",
            "write_mb_s", "compress_ratio", "peak_rss_mb", "peak_traced_kb",
            "n_frames", "zarr_mb", "pff_mb"]
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    typer.echo(f"\nResults written to {out}")
    typer.echo("Load with: import pandas as pd; df = pd.read_csv('bench_results.tsv', sep='\\t')")


if __name__ == "__main__":
    app()
