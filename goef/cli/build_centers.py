"""``python -m goef.cli.build_centers`` - materialise k-shot center datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..centers import DatasetSpec, build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-build-centers",
        description="Build MVTec/VisA few-shot datasets from GOEF representative images.",
    )
    parser.add_argument("--mvtec_source", type=Path, required=True, help="MVTec-AD root (with per-category train/good)")
    parser.add_argument("--mvtec_clusters", type=Path, required=True, help="GOEF selection output for MVTec")
    parser.add_argument("--visa_source", type=Path, help="VisA 1cls root")
    parser.add_argument("--visa_clusters", type=Path, help="GOEF selection output for VisA")
    parser.add_argument("--output", type=Path, required=True, help="output root, e.g. data/2-shot")
    parser.add_argument("--k", type=int, default=5, help="number of clusters in the selection")
    parser.add_argument("--combination-size", type=int, default=1, help="number of centers per dataset (= shot)")
    parser.add_argument(
        "--materialize",
        choices=("meta", "symlink", "hardlink", "copy"),
        default="symlink",
        help="how to expose source images (meta writes metadata only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-representative-content-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = [DatasetSpec("mvtec", args.mvtec_source, args.mvtec_clusters)]
    if (args.visa_source is None) != (args.visa_clusters is None):
        raise SystemExit("--visa_source and --visa_clusters must be given together")
    if args.visa_source is not None:
        specs.append(DatasetSpec("visa", args.visa_source, args.visa_clusters))
    try:
        build(
            specs,
            args.output,
            k=args.k,
            combination_size=args.combination_size,
            materialize=args.materialize,
            overwrite=args.overwrite,
            check_representative_content=not args.skip_representative_content_check,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
