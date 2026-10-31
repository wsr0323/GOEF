"""Weighted k-shot re-evaluation (Sec. IV, Table II).

For a partition of every category's normal training set into ``n`` clusters, a
k-shot evaluation experiment draws k distinct training images.  GOEF maps every
draw onto the ``q``-shot center dataset of its unique cluster set and weights it
by ``N_tau / C(N, k)`` (:mod:`goef.weights`).  The re-evaluated metric of one
category is therefore

.. math::

    M(c) = \\sum_{\\tau} w_\\tau \\, M_q(c; \\tau)

where ``M_q(c; tau)`` is the metric obtained by running the FS-IAD method on the
q-shot center dataset ``tau`` of category ``c``.  Dataset-level values are
macro-averages over categories and the overall value is the equal-weight mean of
MVTec-AD and VisA, exactly as reported in Table II.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .aggregation import METRICS, category_key, read_center_result
from .weights import WeightRow, cluster_sizes_from_summary

#: ``center_01_02_mvtec.csv`` / ``centers_01_02_mvtec_unified.csv`` ...
_RESULT_NAME = re.compile(
    r"^(?P<prefix>centers?)_(?P<centers>\d{2}(?:_\d{2})*)_(?P<dataset>mvtec|visa)"
    r"(?P<unified>_unified)?\.csv$"
)
DATASETS = ("mvtec", "visa")


def center_dir_name(evaluation_shot: int, centers: str) -> str:
    prefix = "center" if evaluation_shot == 1 else "centers"
    return f"{prefix}_{centers}"


def find_result(results_dir: Path, shot: int, dataset: str, centers: str, evaluation_shot: int) -> Path | None:
    """Locate one center-dataset result CSV, preferring the unified schema."""
    candidates = []
    for prefix in ("centers", "center"):
        name = f"{prefix}_{centers}_{dataset}"
        candidates.append(results_dir / f"{shot}-shot" / f"{name}_unified.csv")
        candidates.append(results_dir / f"{shot}-shot" / f"{name}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def index_results(results_dir: Path) -> dict[tuple[int, str, str], Path]:
    """Map ``(shot, dataset, centers)`` to a result CSV.

    The shot is taken from the parent directory name (``1-shot`` ... ``5-shot``);
    ``_unified`` files win over native ones for the same key.
    """
    index: dict[tuple[int, str, str], Path] = {}
    priority: dict[tuple[int, str, str], int] = {}
    for shot_dir in sorted(results_dir.glob("*-shot")):
        match = re.fullmatch(r"(?P<shot>\d+)-shot", shot_dir.name)
        if match is None or not shot_dir.is_dir():
            continue
        shot = int(match.group("shot"))
        for path in sorted(shot_dir.glob("*.csv")):
            found = _RESULT_NAME.match(path.name)
            if found is None:
                continue
            key = (shot, found.group("dataset"), found.group("centers"))
            rank = 1 if found.group("unified") else 0
            if key not in index or rank > priority[key]:
                index[key] = path
                priority[key] = rank
    return index


def weights_for(
    cluster_sizes: Mapping[str, Sequence[int]],
    shot: int,
    dataset: str,
) -> dict[str, list[WeightRow]]:
    """Per-category weight rows for one dataset and shot."""
    from .weights import category_weights

    return {
        category: category_weights(category, sizes, shot, dataset)
        for category, sizes in sorted(cluster_sizes.items())
    }


@dataclass
class MethodResult:
    method: str
    shot: int
    dataset: str
    category_count: int
    metrics: dict[str, float]
    categories: dict[str, dict[str, float]] = field(default_factory=dict)


def aggregate_method(
    method: str,
    results_dir: Path,
    cluster_sizes: Mapping[str, Sequence[int]],
    shot: int,
    dataset: str,
    index: Mapping[tuple[int, str, str], Path] | None = None,
) -> tuple[list[MethodResult], list[str]]:
    """Weighted per-category and dataset-level result of one method."""
    index = index if index is not None else index_results(results_dir)
    weights = weights_for(cluster_sizes, shot, dataset)
    cache: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
    issues: list[str] = []
    rows: list[dict[str, float]] = []

    for category, category_rows in weights.items():
        category = category_key(category)
        weighted = {metric: 0.0 for metric in METRICS}
        for row in category_rows:
            key = (row.evaluation_shot, row.centers)
            if key not in cache:
                path = index.get((row.evaluation_shot, dataset, row.centers))
                # Results may be stored under the shot they were run for; the
                # evaluation shot of a repeated-center draw is q, and its file
                # lives in the q-shot directory.
                if path is None:
                    for candidate_shot in (row.evaluation_shot, shot):
                        path = find_result(
                            results_dir, candidate_shot, dataset, row.centers, row.evaluation_shot
                        )
                        if path is not None:
                            break
                cache[key] = read_center_result(path) if path is not None else {}
            table = cache[key]
            if category not in table:
                issues.append(
                    f"{method} {shot}-shot {dataset}/{category}: missing "
                    f"{row.evaluation_shot}-shot/{row.centers}"
                )
                break
            for metric in METRICS:
                weighted[metric] += row.weight * table[category][metric]
        else:
            rows.append(weighted)

    if issues:
        return [], issues
    if not rows:
        return [], [f"{method} {shot}-shot {dataset}: no category evaluated"]

    dataset_values = {
        metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS
    }
    return (
        [
            MethodResult(
                method=method,
                shot=shot,
                dataset=dataset,
                category_count=len(rows),
                metrics=dataset_values,
                categories={
                    category: values
                    for category, values in zip(weights, rows)
                },
            )
        ],
        [],
    )


def resolve_summary_path(
    sampling_root: Path,
    dataset: str,
    k: int,
    cluster_seed: int,
    center_root: Path | None = None,
    cluster_method: str = "k_means",
) -> Path:
    """Locate the selection summary the centre datasets were built from.

    A centre dataset records its provenance in ``dataset_info.json``; preferring
    that over ``--cluster_seed`` makes it impossible to weight a 1-shot
    experiment with a partition from a different seed.
    """
    name = "MVTec" if dataset == "mvtec" else "VisA"
    if center_root is not None:
        for info in sorted((center_root).glob(f"*-shot/{dataset}/*/dataset_info.json")):
            try:
                payload = json.loads(info.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            cluster_root = payload.get("cluster_root")
            if cluster_root:
                candidate = Path(cluster_root) / "summary.csv"
                if candidate.is_file():
                    return candidate
    # Two layouts are common: a per-method sub-directory (``goef-sample`` output)
    # and the selection run directly under the sampling root (archived layout).
    candidates = [
        sampling_root / cluster_method / f"{name}_k{k}_seed{cluster_seed}" / "summary.csv",
        sampling_root / f"{name}_k{k}_seed{cluster_seed}" / "summary.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"cannot locate the selection summary; tried:\n  {joined}")


def aggregate_all(
    methods: Mapping[str, Path],
    sampling_root: Path,
    shots: Sequence[int] = (1, 2, 3, 4, 5),
    k: int = 5,
    datasets: Sequence[str] = DATASETS,
    allow_partial: bool = False,
    cluster_seed: int = 1234,
    center_root: Path | None = None,
    cluster_method: str = "k_means",
) -> tuple[list[MethodResult], list[str]]:
    """Aggregate every method, shot and dataset; returns results and issues."""
    sizes = {
        dataset: cluster_sizes_from_summary(
            resolve_summary_path(
                sampling_root, dataset, k, cluster_seed, center_root, cluster_method
            )
        )
        for dataset in datasets
    }
    results: list[MethodResult] = []
    issues: list[str] = []
    for method in sorted(methods):
        base = Path(methods[method])
        results_dir = base / "results" if (base / "results").is_dir() else base
        index = index_results(results_dir)
        for shot in shots:
            for dataset in datasets:
                partial, problems = aggregate_method(
                    method, results_dir, sizes[dataset], shot, dataset, index
                )
                if problems:
                    issues.extend(problems)
                    if not allow_partial:
                        continue
                results.extend(partial)
    return results, issues


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def final_rows(results: Iterable[MethodResult]) -> list[dict[str, object]]:
    """Long table used for the paper's Table II."""
    rows = []
    for result in sorted(results, key=lambda item: (item.shot, item.dataset, item.method)):
        row: dict[str, object] = {
            "Shot": result.shot,
            "Dataset": result.dataset,
            "Method": result.method,
            "Category Count": result.category_count,
        }
        row.update({metric: 100.0 * result.metrics[metric] for metric in METRICS})
        rows.append(row)
    return rows


def category_rows(results: Iterable[MethodResult]) -> list[dict[str, object]]:
    """Per-category weighted metrics (``M(c)`` of Eq. 21)."""
    rows = []
    for result in sorted(results, key=lambda item: (item.shot, item.dataset, item.method)):
        for category, values in sorted(result.categories.items()):
            row: dict[str, object] = {
                "Shot": result.shot,
                "Dataset": result.dataset,
                "Method": result.method,
                "Category": category,
            }
            row.update({metric: 100.0 * values[metric] for metric in METRICS})
            rows.append(row)
    return rows


def overall_ranking(
    results: Iterable[MethodResult],
    shots: Sequence[int] = (1, 2, 4),
) -> list[dict[str, object]]:
    """Average performance metric and average rank number (Fig. 5).

    ``average performance`` is the mean of the four metrics over the reported
    shots and both datasets; ``average rank`` is the mean, over every
    (shot, dataset) table, of the method's average tied rank on the four metrics.
    """
    results = [result for result in results if result.shot in shots]
    if not results:
        raise ValueError(f"no result for shots {tuple(shots)}")

    performance: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for metric in METRICS:
            performance[result.method].append(result.metrics[metric])

    frequency: dict[tuple[int, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for result in results:
        frequency[(result.shot, result.dataset)][result.method] = result.metrics

    ranks: dict[str, list[float]] = defaultdict(list)
    for table in frequency.values():
        methods = sorted(table)
        for metric in METRICS:
            ordered = sorted(methods, key=lambda method: table[method][metric], reverse=True)
            position = 0
            while position < len(ordered):
                score = table[ordered[position]][metric]
                end = position + 1
                while end < len(ordered) and table[ordered[end]][metric] == score:
                    end += 1
                tied = (position + 1 + end) / 2
                for method in ordered[position:end]:
                    ranks[method].append(tied)
                position = end

    rows = []
    for method in sorted(performance):
        rows.append(
            {
                "Method": method,
                "Average Performance Metric": 100.0 * sum(performance[method]) / len(performance[method]),
                "Average Rank Number": sum(ranks[method]) / len(ranks[method]),
                "Tables": len(frequency),
                "Shots": ",".join(str(shot) for shot in shots),
            }
        )
    rows.sort(key=lambda row: (row["Average Rank Number"], row["Method"]))
    return rows


def overall_rows(results: Iterable[MethodResult]) -> list[dict[str, object]]:
    """Equal-weight MVTec-AD / VisA mean per method and shot."""
    grouped: dict[tuple[int, str], list[MethodResult]] = defaultdict(list)
    for result in results:
        grouped[(result.shot, result.method)].append(result)
    rows = []
    for (shot, method), items in sorted(grouped.items()):
        if len(items) != 2:
            continue
        row: dict[str, object] = {
            "Shot": shot,
            "Dataset": "overall",
            "Method": method,
            "Category Count": sum(item.category_count for item in items),
        }
        row.update(
            {
                metric: sum(100.0 * item.metrics[metric] for item in items) / len(items)
                for metric in METRICS
            }
        )
        rows.append(row)
    return rows


def rankings(results: Iterable[MethodResult]) -> list[dict[str, object]]:
    """Per (shot, metric, scope) ranking, matching ``method_metric_rankings.csv``."""
    results = list(results)
    rows = []
    for shot in sorted({result.shot for result in results}):
        for metric in METRICS:
            scopes = list(DATASETS) + ["overall"]
            for scope in scopes:
                if scope == "overall":
                    candidates = [
                        result
                        for result in results
                        if result.shot == shot
                    ]
                    grouped: dict[str, list[MethodResult]] = defaultdict(list)
                    for result in candidates:
                        grouped[result.method].append(result)
                    scores = {
                        method: sum(100.0 * item.metrics[metric] for item in items) / len(items)
                        for method, items in grouped.items()
                        if len(items) == 2
                    }
                else:
                    scores = {
                        result.method: 100.0 * result.metrics[metric]
                        for result in results
                        if result.shot == shot and result.dataset == scope
                    }
                for rank, (method, score) in enumerate(
                    sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1
                ):
                    rows.append(
                        {
                            "Shot": shot,
                            "Metric": metric,
                            "Scope": scope,
                            "Rank": rank,
                            "Method": method,
                            "Score": score,
                        }
                    )
    rows.sort(key=lambda row: (row["Shot"], METRICS.index(row["Metric"]), row["Scope"], row["Rank"]))
    return rows


def rank_frequencies(results: Iterable[MethodResult]) -> list[dict[str, object]]:
    """Rank 1..N counts and the average rank of every method (Fig. 5 input)."""
    results = list(results)
    tables: dict[tuple[int, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for result in results:
        tables[(result.shot, result.dataset)][result.method] = result.metrics
    rows = []
    for (shot, dataset), table in sorted(tables.items()):
        if not table:
            continue
        methods = sorted(table)
        counts: dict[str, dict[int, int]] = {method: defaultdict(int) for method in methods}
        rank_sum = {method: 0.0 for method in methods}
        for metric in METRICS:
            ordered = sorted(methods, key=lambda method: table[method][metric], reverse=True)
            position = 0
            while position < len(ordered):
                score = table[ordered[position]][metric]
                end = position + 1
                while end < len(ordered) and table[ordered[end]][metric] == score:
                    end += 1
                average_tied_rank = (position + 1 + end) / 2
                for method in ordered[position:end]:
                    counts[method][position + 1] += 1
                    rank_sum[method] += average_tied_rank
                position = end
        for method in methods:
            row: dict[str, object] = {"Shot": shot, "Dataset": dataset, "Method": method}
            row.update({f"Rank {rank}": counts[method][rank] for rank in range(1, len(methods) + 1)})
            row["Average Rank"] = rank_sum[method] / len(METRICS)
            rows.append(row)
    rows.sort(key=lambda row: (row["Shot"], row["Dataset"], row["Average Rank"], row["Method"]))
    return rows


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]], precision: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.{precision}f}"
                    if isinstance(row.get(key), float) and math.isfinite(row[key])
                    else row.get(key)
                    for key in columns
                }
            )
    return path
