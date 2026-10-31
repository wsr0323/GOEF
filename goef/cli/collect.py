"""``python -m goef.cli.collect`` - copy the raw result tables into ``results/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..collect import (
    collect_centers,
    collect_one_shot,
    split_combined_one_shot,
    write_manifest,
)


def _one_shot(items: list[str]) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise SystemExit(f"--one_shot expects METHOD=DATASET:PATH, got {item!r}")
        method, rest = item.split("=", 1)
        dataset, path = rest.split(":", 1)
        sources.setdefault(method, {})[dataset] = Path(path)
    return sources


def _pairs(items: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{option} expects NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        result[name] = Path(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-collect",
        description="Collect the per-reference sweeps and the per-centre results that back the paper's tables.",
    )
    parser.add_argument("--output", type=Path, required=True, help="results/ directory of this repository")
    parser.add_argument(
        "--one_shot",
        action="append",
        default=[],
        metavar="METHOD=DATASET:PATH",
        help="complete 1-shot sweep of one method and dataset (repeatable)",
    )
    parser.add_argument(
        "--one_shot_split",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="a single CSV that contains several datasets (split by its `dataset` column)",
    )
    parser.add_argument(
        "--dataset_aliases",
        default="MVTec-AD=mvtec,MVTec=mvtec,VisA=visa,visa=visa",
        help="comma separated ALIAS=canonical pairs used when splitting",
    )
    parser.add_argument(
        "--center",
        action="append",
        default=[],
        metavar="METHOD=DIR",
        help="project directory (or results/ directory) of a re-evaluated method (repeatable)",
    )
    parser.add_argument("--shots", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--manifest", type=Path, help="default: <output>/MANIFEST.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aliases = dict(
        pair.split("=", 1) for pair in args.dataset_aliases.split(",") if "=" in pair
    )
    rows = []
    try:
        rows += collect_one_shot(args.output, _one_shot(args.one_shot))
        for method, path in _pairs(args.one_shot_split, "--one_shot_split").items():
            rows += split_combined_one_shot(path, args.output, method, aliases=aliases)
        rows += collect_centers(args.output, _pairs(args.center, "--center"), tuple(args.shots))
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    manifest = args.manifest or (args.output / "MANIFEST.csv")
    write_manifest(manifest, rows)
    one_shot = sum(1 for row in rows if row.kind == "one_shot")
    centers = sum(1 for row in rows if row.kind == "center")
    print(f"collected {one_shot} one-shot table(s) and {centers} centre result(s)")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
