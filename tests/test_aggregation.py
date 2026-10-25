"""Weighted aggregation and result-table readers."""

from __future__ import annotations

import pytest

from goef.aggregation import category_key, macro_average, weighted_metric
from goef.weights import WeightRow


def rows(weights, shot=1):
    return [
        WeightRow(
            dataset="mvtec",
            category="bottle",
            shot=shot,
            evaluation_shot=shot,
            centers=f"{index + 1:02d}",
            weight=weight,
            count=int(weight * 100),
            total_count=100,
        )
        for index, weight in enumerate(weights)
    ]


def constant(value: float) -> dict[str, float]:
    return {"Image AUROC": value, "Image AUPR": value, "Pixel AUROC": value, "Pixel AUPRO": value}


def test_weighted_metric_is_a_convex_combination():
    payload = {"01": constant(0.9), "02": constant(0.5)}
    values = weighted_metric(rows([0.75, 0.25]), lambda row: payload[row.centers])
    assert values["Image AUROC"] == pytest.approx(0.8)


def test_missing_units_are_dropped_and_renormalised():
    metric = lambda row: None if row.centers == "02" else constant(0.9)  # noqa: E731
    values = weighted_metric(rows([0.5, 0.5]), metric)
    assert values["Image AUROC"] == pytest.approx(0.9)


def test_no_available_unit_raises():
    with pytest.raises(ValueError):
        weighted_metric(rows([1.0]), lambda row: None)


def test_macro_average_weights_categories_equally():
    values = macro_average(
        {
            "a": {"Image AUROC": 1.0, "Image AUPR": 1.0, "Pixel AUROC": 1.0, "Pixel AUPRO": 1.0},
            "b": {"Image AUROC": 0.0, "Image AUPR": 0.0, "Pixel AUROC": 0.0, "Pixel AUPRO": 0.0},
        }
    )
    assert values["Image AUROC"] == pytest.approx(0.5)


def test_category_normalisation():
    assert category_key("metal nut") == category_key("metal_nut")
    assert category_key(" Pipe-Fryum ") == "pipe_fryum"
