"""Eq. (22) arithmetic checked against the paper's own weight tables."""

from __future__ import annotations

import math

import pytest

from goef.weights import category_weights, shot_weights, total_draws

BOTTLE = [54, 37, 57, 24, 37]  # MVTec-AD bottle, k=5 k-means partition
TOTAL = 209


def weights(shot):
    return {row.centers: row.weight for row in category_weights("bottle", BOTTLE, shot)}


def test_one_shot_weights_are_cluster_frequencies():
    result = weights(1)
    assert result["01"] == pytest.approx(54 / TOTAL)
    assert result["04"] == pytest.approx(24 / TOTAL)
    assert sum(result.values()) == pytest.approx(1.0)


def test_two_shot_weights_match_the_published_table():
    result = weights(2)
    assert result["01"] == pytest.approx(1431 / math.comb(TOTAL, 2))      # same centre twice
    assert result["01_02"] == pytest.approx(54 * 37 / math.comb(TOTAL, 2))
    assert sum(result.values()) == pytest.approx(1.0)


def test_four_shot_reuse_of_the_two_shot_dataset():
    # (1,1,1,2), (1,1,2,2) and (1,2,2,2) all map onto the 2-shot dataset 01_02
    expected = (
        math.comb(54, 3) * 37 + math.comb(54, 2) * math.comb(37, 2) + 54 * math.comb(37, 3)
    )
    result = weights(4)
    assert result["01_02"] == pytest.approx(expected / total_draws(BOTTLE, 4))
    assert sum(result.values()) == pytest.approx(1.0)


def test_every_shot_sums_to_one():
    for shot in range(1, 6):
        assert sum(row.weight for row in category_weights("bottle", BOTTLE, shot)) == pytest.approx(1.0)


def test_zero_count_combinations_are_omitted():
    sizes = [1, 5]
    # a 2-shot draw cannot take two images from the singleton cluster
    assert [centers for centers, _ in shot_weights(sizes, 2)] == [(1,), (0, 1)]
    assert sum(row.weight for row in category_weights("c", sizes, 2)) == pytest.approx(1.0)


def test_shot_larger_than_training_set_is_rejected():
    with pytest.raises(ValueError):
        total_draws([1, 1], 3)


def test_exact_integer_counts():
    # N_tau must stay an exact integer, never a float
    for _, count in shot_weights(BOTTLE, 3):
        assert isinstance(count, int)
        assert count > 0
