"""Globality-preserving metric aggregation (Eq. 2, Eq. 21, Algo. 1 and 2).

Two aggregation problems share the same arithmetic:

* **1-shot validation** (Sec. III-D): a strategy yields ``n`` representative
  images with cluster sizes ``N_j``; the approximated metric is
  ``mu_r = sum_j N_j P1(I_bar_j) / N`` (Eq. 2).  Here every evaluation unit is a
  single reference image.
* **k-shot re-evaluation** (Sec. IV): a strategy yields a partition into ``n``
  clusters; every k-shot draw is mapped onto the ``q``-shot center dataset of
  its unique cluster set with weight ``N_tau / C(N, k)`` (Eq. 21).  Here every
  evaluation unit is a center dataset.

Both reduce to a weighted average over *evaluation units*, which is what
:func:`weighted_metric` computes.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .weights import WeightRow

METRICS = ("Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO")


def category_key(name: str) -> str:
    """Normalise the category spellings used by the different projects.

    Some adapters write ``metal nut`` where MVTec-AD uses ``metal_nut`` and
    ``pipe fryum`` for VisA's ``pipe_fryum``; the key is only used for joining.
    """
    return re.sub(r"[\s-]+", "_", str(name).strip().lower())


# --------------------------------------------------------------------------- #
# Metric table readers
# --------------------------------------------------------------------------- #
def _metric_values(row: Mapping[str, str], context: str) -> dict[str, float]:
    values = {}
    for metric in METRICS:
        for key, value in row.items():
            if key.strip().startswith(metric):
                try:
                    values[metric] = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{context}: invalid {metric}: {value!r}") from error
                break
        else:
            raise ValueError(f"{context}: missing {metric}")
    return values


def read_per_reference_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Read a one-reference-image sweep CSV.

    Expected columns (order does not matter): ``object``, ``ref_sample`` and the
    four metrics.  This is the schema written by every evaluation adapter under
    ``modifications/``.
    """
    table: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = {"object", "ref_sample"}.difference(columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            key = (row["object"].strip(), row["ref_sample"].strip())
            if key in table:
                raise ValueError(f"{path}: duplicate reference {key}")
            table[key] = _metric_values(row, f"{path}:{key}")
    return table


def read_center_result(path: Path) -> dict[str, dict[str, float]]:
    """Read one center-dataset result CSV (``Category`` + four metrics)."""
    table: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or ()
        category_column = "Category" if "Category" in columns else "category"
        if category_column not in columns:
            raise ValueError(f"{path}: expected a Category column, got {columns}")
        for row in reader:
            category = (row.get(category_column) or "").strip()
            if not category or category.lower() in {"overall", "mean", "average", "all"}:
                continue
            category = category_key(category)
            values = _metric_values(row, f"{path}:{category}")
            if all(math.isfinite(value) for value in values.values()):
                table[category] = values
    return table


# --------------------------------------------------------------------------- #
# Core arithmetic
# --------------------------------------------------------------------------- #
def weighted_metric(
    weights: Sequence[WeightRow],
    lookup: Callable[[WeightRow], Mapping[str, float] | None],
) -> dict[str, float]:
    """Weighted average of the four metrics over the given weights.

    ``lookup`` returns the metrics of one evaluation unit, or ``None`` when the
    unit was not evaluated (then its weight is dropped and the remaining weights
    are renormalised).  Re-normalisation is only used to report partial results;
    :func:`coverage` lets callers refuse incomplete inputs instead.
    """
    total = 0.0
    accumulated = {metric: 0.0 for metric in METRICS}
    for row in weights:
        values = lookup(row)
        if values is None:
            continue
        total += row.weight
        for metric in METRICS:
            accumulated[metric] += row.weight * values[metric]
    if total <= 0:
        raise ValueError("no evaluation unit with a positive weight is available")
    return {metric: accumulated[metric] / total for metric in METRICS}


def coverage(
    weights: Sequence[WeightRow],
    lookup: Callable[[WeightRow], object],
) -> tuple[float, list[WeightRow]]:
    """Weighted fraction of evaluation units that are available."""
    present = 0.0
    missing = []
    for row in weights:
        if lookup(row) is None:
            missing.append(row)
        else:
            present += row.weight
    total = sum(row.weight for row in weights)
    return (present / total if total > 0 else 0.0), missing


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def aggregate_categories(
    weights_by_category: Mapping[str, Sequence[WeightRow]],
    lookup: Callable[[WeightRow], Mapping[str, float] | None],
) -> dict[str, dict[str, float]]:
    """Weighted metrics for every category, keyed by category."""
    return {
        category: weighted_metric(rows, lookup)
        for category, rows in sorted(weights_by_category.items())
    }


def macro_average(category_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Average every metric over categories (the paper's dataset-level value)."""
    if not category_metrics:
        raise ValueError("cannot average an empty set of categories")
    count = len(category_metrics)
    return {
        metric: sum(values[metric] for values in category_metrics.values()) / count
        for metric in METRICS
    }


def group_by_category(weights: Iterable[WeightRow]) -> dict[str, list[WeightRow]]:
    grouped: dict[str, list[WeightRow]] = {}
    for row in weights:
        grouped.setdefault(row.category, []).append(row)
    return grouped
