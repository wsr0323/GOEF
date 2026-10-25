"""End-to-end check of the Table I machinery on a synthetic dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from goef.validation import STRATEGIES, summarize, validate

METRICS = ("Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO")


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture()
def synthetic(tmp_path: Path):
    references = ["a.png", "b.png", "c.png", "d.png"]
    # every reference has the same metric value -> mu_g == mu_r for any weighting
    write_csv(
        tmp_path / "metrics.csv",
        ("object", "ref_sample", *METRICS),
        [("bottle", name, 0.5, 0.5, 0.5, 0.5) for name in references],
    )
    for method, counts, representatives in (
        ("k_means", [2, 2], ["a.png", "c.png"]),
        ("k_medoids", [1, 3], ["b.png", "d.png"]),
        ("k_center", [1, 3], ["a.png", "d.png"]),
        ("coreset", [1, 1], ["b.png", "c.png"]),
        ("random", [1, 1], ["a.png", "b.png"]),
    ):
        write_csv(
            tmp_path / "sampling" / method / "MVTec_k2_seed0" / "summary.csv",
            ("method", "category", "cluster_id", "count", "representative_image"),
            [(method, "bottle", index, count, name) for index, (count, name) in enumerate(zip(counts, representatives))],
        )
    write_csv(
        tmp_path / "sampling" / "random" / "MVTec_k1_seed0" / "summary.csv",
        ("method", "category", "cluster_id", "count", "representative_image"),
        [("random", "bottle", 0, 1, "a.png")],
    )
    return tmp_path


def test_all_strategies_are_exact_for_constant_metrics(synthetic):
    rows = validate(
        {"dummy": {"MVTec": synthetic / "metrics.csv"}},
        synthetic / "sampling",
        k=2,
        seeds=(0,),
    )
    assert len(rows) == len(STRATEGIES) * 4  # six strategies x four metrics
    assert all(row["abs_error"] == 0 for row in rows)
    table = summarize(rows)
    assert {row["Strategy"] for row in table} == set(STRATEGIES)
    assert all(row["epsilon_bar_x100"] == pytest.approx(0.0) for row in table)


def test_missing_selection_is_reported(synthetic):
    with pytest.raises(FileNotFoundError):
        validate(
            {"dummy": {"MVTec": synthetic / "metrics.csv"}},
            synthetic / "sampling",
            k=3,  # no MVTec_k3 runs exist
            seeds=(0,),
        )
