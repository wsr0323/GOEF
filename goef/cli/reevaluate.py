"""``python -m goef.cli.reevaluate`` - reproduce Table II (Sec. IV)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_config
from ..reevaluate import (
    DATASETS,
    aggregate_all,
    category_rows,
    final_rows,
    overall_ranking,
    overall_rows,
    rank_frequencies,
    rankings,
    write_csv,
)


def _methods_from_args(items: list[str]) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--method expects NAME=PROJECT_DIR, got {item!r}")
        name, path = item.split("=", 1)
        methods[name] = Path(path)
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goef-reevaluate",
        description="Aggregate per-center-dataset results with the GOEF k-shot weights.",
    )
    parser.add_argument("--config", type=Path, help="YAML file with sampling_root, shots and methods")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="method name and project directory (repeatable); DIR may be results/ itself",
    )
    parser.add_argument(
        "--sampling_root",
        type=Path,
        help="root holding <Dataset>_k5_seed1234/summary.csv (the partition used to build the centers)",
    )
    parser.add_argument("--shots", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--cluster_method",
        default="k_means",
        help="selection strategy that produced the centres",
    )
    parser.add_argument(
        "--cluster_seed",
        type=int,
        default=1234,
        help="seed of the selection used to build the centres (ignored when --center_root finds dataset_info.json)",
    )
    parser.add_argument(
        "--center_root",
        type=Path,
        help="centre dataset root; its dataset_info.json identifies the exact selection safely",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow_partial", action="store_true", help="keep shots with missing categories")
    parser.add_argument(
        "--figure_shots",
        nargs="*",
        type=int,
        default=[1, 2, 4],
        help="shots averaged by the Fig. 5 summary (default: the three reported settings)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    methods = {name: Path(path) for name, path in (config.get("methods") or {}).items()}
    methods.update(_methods_from_args(args.method))
    if not methods:
        print("ERROR: at least one --method NAME=DIR (or a config) is required", file=sys.stderr)
        return 2
    sampling_root = args.sampling_root or (
        Path(config["sampling_root"]) if config.get("sampling_root") else None
    )
    if sampling_root is None:
        print("ERROR: --sampling_root (or sampling_root in the config) is required", file=sys.stderr)
        return 2
    shots = tuple(args.shots) if args.shots else tuple(config.get("shots", (1, 2, 3, 4, 5)))

    try:
        results, issues = aggregate_all(
            methods,
            sampling_root,
            shots=shots,
            k=args.k,
            datasets=tuple(args.datasets),
            allow_partial=args.allow_partial,
            cluster_seed=args.cluster_seed,
            center_root=args.center_root,
            cluster_method=args.cluster_method,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if issues:
        print(f"{len(issues)} incomplete categories (shots without a full grid are skipped):")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... {len(issues) - 20} more")

    output = args.output
    metrics = ["Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO"]
    final = final_rows(results)
    overall = overall_rows(results)
    write_csv(
        output / "final_weighted_results.csv",
        ["Shot", "Dataset", "Method", "Category Count", *metrics],
        final + overall,
    )
    write_csv(
        output / "all_final_weighted_results.csv",
        ["Shot", "Dataset", "Method", "Category Count", *metrics],
        final,
    )
    write_csv(
        output / "method_metric_rankings.csv",
        ["Shot", "Metric", "Scope", "Rank", "Method", "Score"],
        rankings(results),
    )
    write_csv(
        output / "category_weighted_results.csv",
        ["Shot", "Dataset", "Method", "Category", *metrics],
        category_rows(results),
    )
    write_csv(
        output / "figure5_ranking.csv",
        ["Method", "Average Performance Metric", "Average Rank Number", "Tables", "Shots"],
        overall_ranking(results, tuple(args.figure_shots)),
    )
    frequency = rank_frequencies(results)
    rank_columns = [key for key in (frequency[0] if frequency else {}) if key.startswith("Rank ")]
    write_csv(
        output / "metric_rank_frequency.csv",
        ["Shot", "Dataset", "Method", *rank_columns, "Average Rank"],
        frequency,
    )
    print(f"wrote {len(final)} dataset rows and {len(overall)} overall rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
