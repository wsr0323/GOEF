"""``python -m goef.cli.weights`` - globality weights of Eq. (21)-(22)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..weights import build_weights, cluster_sizes_from_summary, write_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-weights",
        description=(
            "Compute the weight of every k-shot center combination. The weights of "
            "one category sum to one and implement Eq. (22) for arbitrary k."
        ),
    )
    parser.add_argument("--summary", type=Path, required=True, help="summary.csv of a goef-sample run")
    parser.add_argument("--dataset", default="", help="dataset name stored in the output")
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output CSV")
    parser.add_argument("--all_shots", type=int, nargs="*", help="also write these shot values into the same file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sizes = cluster_sizes_from_summary(args.summary)
        shots = [args.shot] + list(args.all_shots or [])
        rows = []
        for shot in sorted(dict.fromkeys(shots)):
            rows.extend(build_weights(sizes, shot, args.dataset))
        write_weights(args.output, rows)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} weight rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
