"""Empirical validation of evaluation strategies (Sec. III-D, Table I).

For every (dataset, detector, category, metric, seed) the validation computes

* the ideal globality-preserving metric ``mu_g`` -- the mean performance obtained
  when *every* normal training image is used as the single reference, and
* the approximate metric ``mu_r`` of each evaluation strategy (SINGLE, RE-RS,
  RE-CS, GOEF-KC, GOEF-KD, GOEF-KM),

and then reports the approximation error ``eps = |mu_g - mu_r|`` together with
the per-experiment rank ``r`` of ``eps`` among the six strategies.

Two averages are reported, because they answer different questions:

``epsilon_bar_x100``
    Table I of the paper.  The error is first averaged over the four metrics and
    all categories of one (dataset, detector, seed) table, then averaged over the
    50 tables (5 seeds x 5 detectors x 2 datasets).  Every table contributes
    equally.
``epsilon_bar_pooled_x100``
    The error averaged over all 2700 individual comparisons.  Because MVTec-AD
    contributes 15 categories per table and VisA 12, this slightly over-weights
    MVTec-AD relative to the table-level average.

``rank_bar``
    The average competition rank of ``eps`` within each
    (dataset, detector, seed, category, metric) experiment.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .aggregation import METRICS, _metric_values, read_per_reference_metrics, weighted_metric
from .weights import WeightRow, centers_label

#: paper name -> directory name produced by ``goef-sample``
STRATEGIES: dict[str, str] = {
    "SINGLE": "single",
    "RE-RS": "random",
    "RE-CS": "coreset",
    "GOEF-KC": "k_center",
    "GOEF-KD": "k_medoids",
    "GOEF-KM": "k_means",
}


@dataclass(frozen=True)
class RunSelection:
    """One representative-selection run of one category."""

    method: str
    category: str
    representatives: list[str]
    counts: list[int]
    source: Path

    def weights(self) -> list[WeightRow]:
        # RE-RS / RE-CS draw n independent references; every draw has weight 1/n.
        counts = [1] * len(self.representatives) if self.method in {"random", "coreset"} else self.counts
        total = sum(counts)
        if total <= 0:
            raise ValueError(f"{self.source}: non-positive cluster size total")
        return [
            WeightRow(
                dataset="",
                category=self.category,
                shot=1,
                evaluation_shot=1,
                centers=centers_label([cluster_id]),
                weight=count / total,
                count=count,
                total_count=total,
            )
            for cluster_id, count in enumerate(counts)
        ]


# --------------------------------------------------------------------------- #
# Reading selections
# --------------------------------------------------------------------------- #
def read_summary(path: Path, method: str) -> dict[str, RunSelection]:
    """Read one ``summary.csv`` into ``category -> RunSelection``."""
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"category", "cluster_id", "count", "representative_image"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        for row in reader:
            grouped[row["category"].strip()].append(
                (int(row["cluster_id"]), int(row["count"]), row["representative_image"].strip())
            )
    return {
        category: RunSelection(
            method=method,
            category=category,
            representatives=[name for _, _, name in sorted(entries)],
            counts=[count for _, count, _ in sorted(entries)],
            source=path,
        )
        for category, entries in grouped.items()
    }


def find_run(root: Path, method: str, dataset: str, k: int, seed: int) -> Path:
    path = root / method / f"{dataset}_k{k}_seed{seed}" / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing sampling run: {path}")
    return path


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def actual_metrics(table: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """``mu_g``: mean over every single-reference evaluation of a category."""
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for (category, _reference), values in table.items():
        grouped[category].append(values)
    return {
        category: {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        for category, rows in grouped.items()
    }


def validate_dataset(
    detector: str,
    table: Mapping[tuple[str, str], Mapping[str, float]],
    dataset: str,
    root: Path,
    k: int,
    seeds: Sequence[int],
    strategies: Mapping[str, str] = STRATEGIES,
    single_method: str = "random",
    single_k: int = 1,
    legacy_single_precision: bool = False,
) -> list[dict[str, object]]:
    """Return one row per (seed, strategy, category, metric)."""
    actual = actual_metrics(table)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        selections = {}
        for strategy, method in strategies.items():
            if method == "single":
                path = find_run(root, single_method, dataset, single_k, seed)
            else:
                path = find_run(root, method, dataset, k, seed)
            selections[strategy] = read_summary(path, method)

        categories = sorted(actual)
        for strategy in strategies:
            for category in categories:
                selection = selections[strategy].get(category)
                if selection is None:
                    raise ValueError(f"{detector}/{category}: {strategy} has no representative")
                by_center = {
                    centers_label([cluster_id]): name
                    for cluster_id, name in enumerate(selection.representatives)
                }
                estimate = weighted_metric(
                    selection.weights(),
                    lambda row, s=selection, m=by_center: table.get((s.category, m[row.centers])),
                )
                for metric in METRICS:
                    # The paper's tables store metrics with six decimals; taking
                    # the difference at that precision is what the archived
                    # pipeline does and keeps saturated metrics (1.0) tied.
                    estimate_value = estimate[metric]
                    actual_value = actual[category][metric]
                    # The paper's tables store metrics with six decimals and take
                    # the difference in exact decimal arithmetic; doing the same
                    # here keeps the ties that the rank metric depends on.
                    if legacy_single_precision and strategy == "SINGLE":
                        # The archived pipeline copied SINGLE from the raw metric
                        # CSV (full precision) while every other strategy was
                        # stored with six decimals.  Reproducing that asymmetry
                        # reproduces the published SINGLE rank exactly; it is off
                        # by default because it slightly disadvantages SINGLE.
                        error = abs(Decimal(repr(float(estimate_value))) - Decimal(f"{actual_value:.6f}"))
                    else:
                        error = abs(Decimal(f"{estimate_value:.6f}") - Decimal(f"{actual_value:.6f}"))
                    rows.append(
                        {
                            "dataset": dataset,
                            "detector": detector,
                            "seed": seed,
                            "strategy": strategy,
                            "category": category,
                            "metric": metric,
                            "estimate": estimate_value,
                            "actual": actual_value,
                            "abs_error": float(error),
                            "_error_key": error,
                        }
                    )
    return rows


def load_detector_table(spec: object) -> dict[tuple[str, str], dict[str, float]]:
    """Load one detector CSV.

    *spec* is either a path or a mapping ``{"path": ..., "filter": [...]}``; the
    optional filter keeps only rows whose ``dataset`` column matches one of the
    given values (needed for combined MVTec+VisA tables such as INP-Former's).
    """
    if isinstance(spec, (str, Path)):
        return read_per_reference_metrics(Path(spec))
    if not isinstance(spec, Mapping):
        raise TypeError(f"unsupported detector specification: {spec!r}")
    path = Path(spec["path"])
    filters = {str(value).casefold() for value in spec.get("filter", [])}
    if not filters:
        return read_per_reference_metrics(path)

    table: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            if dataset.casefold() not in filters:
                continue
            key = (row["object"].strip(), row["ref_sample"].strip())
            if key in table:
                raise ValueError(f"{path}: duplicate reference {key}")
            table[key] = _metric_values(row, f"{path}:{key}")
    if not table:
        raise ValueError(f"{path}: dataset filter {sorted(filters)} matched no row")
    return table


def validate(
    detectors: Mapping[str, Mapping[str, object]],
    root: Path,
    k: int = 5,
    seeds: Sequence[int] = (0, 1, 42, 1234, 3407),
    strategies: Mapping[str, str] = STRATEGIES,
    legacy_single_precision: bool = False,
) -> list[dict[str, object]]:
    """Run the validation for ``detector -> dataset -> metric-table spec``."""
    rows: list[dict[str, object]] = []
    for detector in sorted(detectors):
        for dataset in sorted(detectors[detector]):
            table = load_detector_table(detectors[detector][dataset])
            print(
                f"validating {detector}/{dataset}: {len(table)} reference evaluations, "
                f"{len(actual_metrics(table))} categories"
            )
            rows.extend(
                validate_dataset(
                    detector,
                    table,
                    dataset,
                    root,
                    k,
                    list(seeds),
                    strategies=strategies,
                    legacy_single_precision=legacy_single_precision,
                )
            )
    add_ranks(rows)
    return rows


def add_ranks(rows: list[dict[str, object]]) -> None:
    """Competition ranks of ``abs_error`` inside each experiment (in place)."""
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["detector"], row["seed"], row["category"], row["metric"])
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["_error_key"], row["strategy"]))
        previous_error = None
        previous_rank = 0
        for position, row in enumerate(group, start=1):
            error = row["_error_key"]
            if previous_error is not None and error == previous_error:
                row["rank"] = previous_rank
            else:
                row["rank"] = position
                previous_error, previous_rank = error, position


def _table_macro(rows: Sequence[Mapping[str, object]], metrics: Sequence[str]) -> float:
    """Mean |error| over the categories and *metrics* of one table."""
    selected = [row for row in rows if row["metric"] in metrics]
    return sum(float(row["abs_error"]) for row in selected) / len(selected)


def summarize(
    rows: Iterable[Mapping[str, object]],
    strategies: Sequence[str] = tuple(STRATEGIES),
) -> list[dict[str, object]]:
    """Table I: ``eps_bar * 100`` and ``r_bar`` per strategy and metric scope."""
    rows = list(rows)
    scopes: list[tuple[str, tuple[str, ...]]] = [("ALL", METRICS)] + [
        (metric, (metric,)) for metric in METRICS
    ]
    table = []
    for scope_name, metrics in scopes:
        for strategy in strategies:
            selected = [
                row for row in rows if row["strategy"] == strategy and row["metric"] in metrics
            ]
            if not selected:
                continue
            tables: dict[tuple, list[Mapping[str, object]]] = defaultdict(list)
            for row in selected:
                tables[(row["dataset"], row["detector"], row["seed"])].append(row)
            per_table = [_table_macro(items, metrics) for items in tables.values()]
            errors = [float(row["abs_error"]) for row in selected]
            ranks = [float(row["rank"]) for row in selected]
            table.append(
                {
                    "Metric": scope_name,
                    "Strategy": strategy,
                    "epsilon_bar_x100": 100.0 * sum(per_table) / len(per_table),
                    "epsilon_bar_pooled_x100": 100.0 * sum(errors) / len(errors),
                    "rank_bar": sum(ranks) / len(ranks),
                    "experiments": len(selected),
                }
            )
    scope_order = {name: index for index, (name, _) in enumerate(scopes)}
    table.sort(key=lambda row: (scope_order[row["Metric"]], row["epsilon_bar_x100"]))
    return table


DETAIL_COLUMNS = (
    "dataset",
    "detector",
    "seed",
    "strategy",
    "category",
    "metric",
    "estimate",
    "actual",
    "abs_error",
    "rank",
)
