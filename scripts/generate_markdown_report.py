#!/usr/bin/env python3
import argparse
from pathlib import Path

import xarray as xr


def build_report(outdir: Path) -> str:
    lines = ["# PANOSETI Pipeline Quality Report\n"]

    for level in ["L1", "L2"]:
        stores_dir = outdir / level
        quicklooks_dir = outdir / f"{level}_quicklooks"

        if not stores_dir.exists():
            continue

        lines.append(f"## {level} Products\n")

        for zarr_path in sorted(stores_dir.glob("*.zarr")):
            lines.append(f"### {zarr_path.name}\n")

            try:
                ds = xr.open_zarr(zarr_path, consolidated=False)
                qc = ds.attrs.get("qc", {})
                if qc:
                    status = "🟩 PASS" if qc.get("isgood") else "🟥 FAIL"
                    lines.append(f"**QC Status:** {status}\n")
                    lines.append("| Check | Threshold | Value | Passed |")
                    lines.append("|---|---|---|---|")
                    for chk in qc.get("checks", []):
                        icon = "🟢" if chk.get("passed") else "🔴"
                        lines.append(
                            f"| `{chk['key']}` | {chk.get('threshold', '')} | {chk.get('value', '')} | {icon} |"
                        )
                    lines.append("\n")
                else:
                    lines.append("*No QC metadata found in Zarr attrs.*\n")
            except Exception as e:
                lines.append(f"*Error reading zarr metadata: {e}*\n")

            # Embed image if exists
            if quicklooks_dir.exists():
                # Let's search for a file that contains the core name
                core_name = zarr_path.name.replace(f".{level}.zarr", "").replace(".zarr", "")
                pngs = list(quicklooks_dir.glob(f"*{core_name}*.png"))
                if pngs:
                    png_path = pngs[0].resolve()
                    lines.append(f"![Quicklook for {core_name}]({png_path})\n")

            lines.append("---\n")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("overview.md"))
    args = parser.parse_args()

    report = build_report(args.outdir)
    args.output.write_text(report)
    print(f"Wrote report to {args.output}")
