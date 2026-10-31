"""Fig. 5 aggregation and per-category weighted output."""

from __future__ import annotations

import pytest

from goef.reevaluate import MethodResult, category_rows, overall_ranking

METRICS = ("Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO")


def result(method, shot, dataset, value):
    metrics = {metric: value for metric in METRICS}
    return MethodResult(
        method=method,
        shot=shot,
        dataset=dataset,
        category_count=1,
        metrics=metrics,
        categories={"bottle": metrics},
    )


def test_overall_ranking_uses_the_selected_shots():
    results = [
        result("A", 1, "mvtec", 0.9),
        result("B", 1, "mvtec", 0.8),
        result("A", 2, "mvtec", 0.9),
        result("B", 2, "mvtec", 0.7),
        result("A", 4, "mvtec", 0.9),
        result("B", 4, "mvtec", 0.6),
        result("A", 3, "mvtec", 0.0),   # not reported -> ignored
        result("B", 3, "mvtec", 1.0),
    ]
    rows = overall_ranking(results, shots=(1, 2, 4))
    by_method = {row["Method"]: row for row in rows}
    assert by_method["A"]["Average Rank Number"] == pytest.approx(1.0)
    assert by_method["B"]["Average Rank Number"] == pytest.approx(2.0)
    assert by_method["A"]["Average Performance Metric"] == pytest.approx(90.0)
    assert by_method["B"]["Average Performance Metric"] == pytest.approx(70.0)
    assert by_method["A"]["Tables"] == 3


def test_tied_scores_share_the_average_rank():
    results = [
        result("A", 1, "mvtec", 0.5),
        result("B", 1, "mvtec", 0.5),
        result("C", 1, "mvtec", 0.1),
    ]
    rows = {row["Method"]: row for row in overall_ranking(results, shots=(1,))}
    assert rows["A"]["Average Rank Number"] == pytest.approx(1.5)
    assert rows["B"]["Average Rank Number"] == pytest.approx(1.5)
    assert rows["C"]["Average Rank Number"] == pytest.approx(3.0)


def test_category_rows_are_percentages():
    rows = category_rows([result("A", 1, "mvtec", 0.5)])
    assert len(rows) == 1
    assert rows[0]["Category"] == "bottle"
    assert rows[0]["Image AUROC"] == pytest.approx(50.0)


def test_empty_selection_raises():
    with pytest.raises(ValueError):
        overall_ranking([result("A", 5, "mvtec", 0.5)], shots=(1, 2, 4))
