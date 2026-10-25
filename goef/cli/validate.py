"""``python -m goef.cli.validate`` - reproduce Table I (Sec. III-D)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ..config import expand_path, load_config
from ..validation import DETAIL_COLUMNS, STRATEGIES, summarize, validate


def _detectors_from_config(config: dict) -> dict[str, dict[str, object]]:
    detectors: dict[str, dict[str, object]] = {}
    for dataset, values in (config.get("datasets") or {}).items():
        for name, spec in (values.get("detectors") or {}).items():
            detectors.setdefault(name, {})[dataset] = expand_path(spec) if isinstance(spec, str) else spec
    return detectors


def _detectors_from_args(items: list[str]) -> dict[str, dict[str, object]]:
    """Parse ``--detector NAME=DATASET:PATH[::FILTER,FILTER]``."""
    detectors: dict[str, dict[str, object]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise SystemExit(f"--detector expects NAME=DATASET:PATH, got {item!r}")
        name, rest = item.split("=", 1)
        dataset, path = rest.split(":", 1)
        spec: object = path
        if "::" in path:
            path, raw_filter = path.split("::", 1)
            spec = {"path": path, "filter": [value for value in raw_filter.split(",") if value]}
        detectors.setdefault(name, {})[dataset] = spec
    return detectors


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]], precision: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.{precision}f}" if isinstance(row.get(key), float) else row.get(key)
                    for key in columns
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-validate",
        description="Compare SINGLE / RE-RS / RE-CS / GOEF-KC / GOEF-KD / GOEF-KM against the ideal mu_g.",
    )
    parser.add_argument("--config", type=Path, help="YAML file with sampling_root, seeds and detector tables")
    parser.add_argument(
        "--detector",
        action="append",
        default=[],
        metavar="NAME=DATASET:PATH[::FILTER]",
        help="add one per-reference metric table (repeatable)",
    )
    parser.add_argument("--sampling_root", type=Path, help="root containing <method>/<Dataset>_k5_seed<seed>/summary.csv")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=list(STRATEGIES),
        help="restrict the comparison to a subset (default: all six)",
    )
    parser.add_argument(
        "--legacy_single_precision",
        action="store_true",
        help=(
            "reproduce the published SINGLE row exactly: the archived pipeline kept full "
            "precision for SINGLE while storing the other strategies with six decimals"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    detectors = _detectors_from_config(config)
    detectors.update(_detectors_from_args(args.detector))
    sampling_root = args.sampling_root or (
        Path(config["sampling_root"]) if config.get("sampling_root") else None
    )
    seeds = tuple(args.seeds if args.seeds is not None else config.get("seeds", (0, 1, 42, 1234, 3407)))
    k = args.k if args.k is not None else int(config.get("k", 5))

    if sampling_root is None:
        print("ERROR: --sampling_root (or sampling_root in the config) is required", file=sys.stderr)
        return 2
    if not detectors:
        print("ERROR: no detector table given (--detector or config)", file=sys.stderr)
        return 2
    if not sampling_root.is_dir():
        print(f"ERROR: sampling root not found: {sampling_root}", file=sys.stderr)
        return 2

    try:
        rows = validate(
            detectors,
            sampling_root,
            k=k,
            seeds=seeds,
            strategies={name: STRATEGIES[name] for name in (args.strategies or STRATEGIES)},
            legacy_single_precision=args.legacy_single_precision,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    output = args.output
    _write_csv(output / "validation_details.csv", list(DETAIL_COLUMNS), rows)
    table = summarize(rows)
    _write_csv(
        output / "table_validation.csv",
        ["Metric", "Strategy", "epsilon_bar_x100", "epsilon_bar_pooled_x100", "rank_bar", "experiments"],
        table,
    )
    print(f"\nwrote {output / 'validation_details.csv'} ({len(rows)} comparisons)")
    print(f"wrote {output / 'table_validation.csv'}")
    print("\nTable I (eps_bar x100 / rank_bar)")
    for row in table:
        if row["Metric"] != "ALL":
            continue
        print(
            f"  {row['Strategy']:8s} eps={row['epsilon_bar_x100']:.3f} "
            f"(pooled {row['epsilon_bar_pooled_x100']:.3f})  rank={row['rank_bar']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
