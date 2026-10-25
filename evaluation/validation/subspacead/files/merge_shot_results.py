#!/usr/bin/env python
"""把每个 reference 的单行 benchmark_results.csv 合并成一张总表。

输出列与用户期望格式完全一致:
    object, Run, ref_sample, 参考索引, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO
"""

import argparse
import glob
import os
import sys

import pandas as pd


COLUMNS = [
    "object",
    "Run",
    "ref_sample",
    "参考索引",
    "Image AUROC",
    "Image AUPR",
    "Pixel AUROC",
    "Pixel AUPRO",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge per-run results into one table.")
    ap.add_argument("--root", required=True, help="OUT_ROOT passed to the bash script.")
    ap.add_argument("--out", default=None, help="Output CSV path.")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[merge] root not found: {args.root}", file=sys.stderr)
        return 1

    pattern = os.path.join(args.root, "*", "*", "*", "benchmark_results.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[merge] no benchmark_results.csv found under {args.root}", file=sys.stderr)
        return 1

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[merge] skip {p}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        if "object" in df.columns:
            df = df[df["object"] != "Average"]
        frames.append(df)

    if not frames:
        print("[merge] no valid rows collected", file=sys.stderr)
        return 1

    out_df = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        if c not in out_df.columns:
            out_df[c] = pd.NA
    out_df = out_df[COLUMNS]

    out_df["参考索引"] = pd.to_numeric(out_df["参考索引"], errors="coerce")
    out_df = out_df.sort_values(
        by=["object", "参考索引"], ascending=[True, True], kind="stable"
    ).reset_index(drop=True)

    out_path = args.out or os.path.join(args.root, "all_runs_table.csv")
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"[merge] wrote {len(out_df)} rows -> {out_path}")
    print(out_df.groupby("object").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())