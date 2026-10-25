"""``python -m goef.cli.sample`` - select GOEF representative images."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..features import DEFAULT_MODEL
from ..sampling import run
from ..selection import METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-sample",
        description=(
            "Select normal reference images with a GOEF clustering strategy "
            "(k-means / k-medoids / k-center), PatchCore coreset, or random sampling."
        ),
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_dir", type=Path, help="a single category image directory")
    source.add_argument("--dataset_root", type=Path, help="dataset root with one directory per category")
    parser.add_argument("--sub_path", default="train/good", help="image path inside a category")
    parser.add_argument("--categories", help="comma separated category whitelist")
    parser.add_argument("--out_dir", type=Path, required=True, help="output directory of this run")
    parser.add_argument("--k", type=int, default=5, help="number of representatives (0 = silhouette search)")
    parser.add_argument("--max_k", type=int, default=10, help="upper bound when --k is 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, help="DINOv3 checkpoint (not needed for --method random)")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--coreset_dim", type=int, default=128, help="PatchCore random projection dim (0 = off)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--copy_images", action="store_true", help="copy representatives and clustered images")
    parser.add_argument("--preview", action="store_true", help="write clusters_preview.jpg (clustering methods)")
    parser.add_argument("--thumb_size", type=int, default=160)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--no_recursive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
