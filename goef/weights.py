"""Globality weights for arbitrary k-shot evaluation (Eq. 21-22, Algo. 2).

Given a partition of the normal training set into ``n`` clusters with sizes
``N_1 .. N_n`` (``N = sum_j N_j``), the ideal k-shot metric of Eq. (17) averages
the performance function ``P_k`` over *all* ordered k-tuples of distinct training
images.  GOEF replaces every image by the representative of its cluster, so a
k-tuple is fully described by the multiset of cluster indices it draws from.

Two relaxation strategies make the sum tractable (Sec. III-F):

* the order of the k reference images does not matter, and
* a repeated representative carries no extra information, so ``P_k`` of a
  k-tuple with ``q`` distinct clusters equals the q-shot performance
  ``P_q`` of those clusters.

Consequently all k-tuples that share a unique cluster set ``tau`` (``|tau| = q``)
collapse onto a single evaluation of the q-shot center dataset, and

.. math::

    N_\\tau = \\sum_{a_1 + ... + a_q = k} \\prod_{s=1}^{q} \\binom{N_{\\tau_s}}{a_s}

    w_\\tau = N_\\tau / \\binom{N}{k}

which is Eq. (22) after cancelling the ``k!`` factor that appears on both sides
of the ordered formulation.  The weights of one category sum to one, so the
aggregation in :mod:`goef.aggregation` is a proper weighted average.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class WeightRow:
    dataset: str
    category: str
    shot: int
    evaluation_shot: int
    centers: str
    weight: float
    count: int
    total_count: int


def centers_label(cluster_ids: Sequence[int]) -> str:
    """``(0, 2)`` -> ``'01_03'`` (1-based, zero padded, sorted)."""
    return "_".join(f"{int(cluster_id) + 1:02d}" for cluster_id in cluster_ids)


def cluster_sizes_from_summary(summary_path: Path) -> dict[str, list[int]]:
    """Read ``category -> [N_0, N_1, ...]`` from a ``goef-sample`` summary.csv."""
    sizes: dict[str, dict[int, int]] = {}
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"category", "cluster_id", "count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{summary_path}: expected columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            category = row["category"].strip()
            cluster_id = int(row["cluster_id"])
            bucket = sizes.setdefault(category, {})
            if cluster_id in bucket:
                raise ValueError(f"{summary_path}: duplicate {category}/{cluster_id}")
            bucket[cluster_id] = int(row["count"])
    return {
        category: [values[index] for index in sorted(values)]
        for category, values in sizes.items()
    }


def shot_weights(cluster_sizes: Sequence[int], shot: int) -> list[tuple[tuple[int, ...], int]]:
    """All nonzero ``(unique cluster set, count)`` pairs for one category.

    ``count`` is ``N_tau`` of Eq. (22); the denominator ``C(N, k)`` is returned
    by :func:`total_draws` so callers can keep exact integers.
    """
    if shot <= 0:
        raise ValueError("shot must be positive")
    sizes = [int(value) for value in cluster_sizes]
    if any(value < 0 for value in sizes):
        raise ValueError("cluster sizes must be non-negative")
    total = sum(sizes)
    if shot > total:
        raise ValueError(
            f"cannot draw {shot} distinct images from {total} training images"
        )

    grouped: dict[tuple[int, ...], int] = {}
    for combination in itertools.combinations_with_replacement(range(len(sizes)), shot):
        multiplicities = {index: combination.count(index) for index in set(combination)}
        count = 1
        for index, multiplicity in multiplicities.items():
            count *= math.comb(sizes[index], multiplicity)
        if count == 0:
            continue
        key = tuple(sorted(multiplicities))
        grouped[key] = grouped.get(key, 0) + count
    return sorted(grouped.items(), key=lambda item: (len(item[0]), item[0]))


def total_draws(cluster_sizes: Sequence[int], shot: int) -> int:
    """``C(N, k)``: the number of unordered k-subsets of the training set."""
    total = sum(int(value) for value in cluster_sizes)
    if shot > total:
        raise ValueError(f"cannot draw {shot} distinct images from {total} training images")
    return math.comb(total, shot)


def category_weights(
    category: str,
    cluster_sizes: Sequence[int],
    shot: int,
    dataset: str = "",
) -> list[WeightRow]:
    """Weighted evaluation plan of one category for a given shot."""
    total = total_draws(cluster_sizes, shot)
    rows = []
    for centers, count in shot_weights(cluster_sizes, shot):
        rows.append(
            WeightRow(
                dataset=dataset,
                category=category,
                shot=shot,
                evaluation_shot=len(centers),
                centers=centers_label(centers),
                weight=count / total,
                count=count,
                total_count=total,
            )
        )
    check = sum(row.weight for row in rows)
    if not math.isclose(check, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{category}: weights sum to {check}, expected 1")
    return rows


def build_weights(
    cluster_sizes: Mapping[str, Sequence[int]],
    shot: int,
    dataset: str = "",
) -> list[WeightRow]:
    """Weights for every category of one dataset."""
    rows: list[WeightRow] = []
    for category in sorted(cluster_sizes):
        rows.extend(category_weights(category, cluster_sizes[category], shot, dataset))
    return rows


WEIGHT_COLUMNS = (
    "dataset",
    "category",
    "shot",
    "evaluation_shot",
    "centers",
    "count",
    "total_count",
    "weight",
)


def write_weights(path: Path, rows: Iterable[WeightRow]) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WEIGHT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "category": row.category,
                    "shot": row.shot,
                    "evaluation_shot": row.evaluation_shot,
                    "centers": row.centers,
                    "count": row.count,
                    "total_count": row.total_count,
                    "weight": f"{row.weight:.12f}",
                }
            )
    return path


def read_weights(path: Path) -> list[WeightRow]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                WeightRow(
                    dataset=row["dataset"],
                    category=row["category"],
                    shot=int(row["shot"]),
                    evaluation_shot=int(row["evaluation_shot"]),
                    centers=row["centers"],
                    weight=float(row["weight"]),
                    count=int(row["count"]),
                    total_count=int(row["total_count"]),
                )
            )
    return rows
