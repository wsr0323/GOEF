"""Reference-selection strategies compared in Sec. III-D.

Five strategies are supported.  They all return *indices into the training
image list*:

``k_means``    K-means clusters; the member closest to each centroid is chosen
               (the "small trick" of Sec. III-C makes a real image the
               representative).  This is GOEF-KM in the paper.
``k_medoids``  FastPAM1 K-medoids; the medoid of each cluster is the
               representative.  This is GOEF-KD in the paper.
``k_center``   Farthest-first K-center; the centers are the representatives.
               This is GOEF-KC, included as the *negative* control because it
               favours outliers instead of compact clusters.
``coreset``    PatchCore's greedy coreset over randomly projected features.
               Used as the RE-CS repeated-experiment sampler.
``random``     Uniform random sampling without replacement.  Used as RE-RS.

The k-means / k-medoids implementations optimise exactly the objectives in
Eq. (6) and Eq. (7) of the paper, i.e. the upper bound of the globality
approximation error in Eq. (5).
"""

from __future__ import annotations

import itertools
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .features import patchcore_random_projection

METHODS = ("coreset", "k_center", "k_means", "k_medoids", "random")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def assign_to_centers(features: np.ndarray, center_indices: np.ndarray) -> np.ndarray:
    """Label every sample with its nearest representative."""
    distances = pairwise_distances(features, features[center_indices], metric="euclidean")
    return distances.argmin(axis=1).astype(int)


def farthest_first(features: np.ndarray, k: int, first_index: int) -> np.ndarray:
    """Greedy K-center: repeatedly add the point farthest from the current set."""
    centers = [int(first_index)]
    min_distances = np.linalg.norm(features - features[first_index], axis=1)
    while len(centers) < k:
        min_distances[centers] = -math.inf
        next_index = int(np.argmax(min_distances))
        centers.append(next_index)
        distances = np.linalg.norm(features - features[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return np.asarray(centers, dtype=int)


# --------------------------------------------------------------------------- #
# coreset (PatchCore)
# --------------------------------------------------------------------------- #
def run_coreset(
    features: np.ndarray,
    k: int,
    seed: int = 0,
    target_dim: int = 128,
) -> np.ndarray:
    """Select exactly *k* representatives with PatchCore's greedy rule.

    PatchCore initialises the coreset with the sample whose distance vector has
    the largest L2 norm and then repeatedly adds the sample that is farthest
    from the already selected set.  ``target_dim`` reproduces PatchCore's
    optional random linear reduction (``0`` disables it).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {features.shape}")
    k = min(max(int(k), 1), len(features))
    reduced = patchcore_random_projection(features, seed, target_dim)
    distance_matrix = pairwise_distances(reduced, metric="euclidean")
    anchor_distances = np.linalg.norm(distance_matrix, axis=1)
    representatives = []
    for _ in range(k):
        select_index = int(np.argmax(anchor_distances))
        representatives.append(select_index)
        anchor_distances = np.minimum(anchor_distances, distance_matrix[:, select_index])
    return np.asarray(representatives, dtype=int)


# --------------------------------------------------------------------------- #
# K-center
# --------------------------------------------------------------------------- #
def run_k_center(features: np.ndarray, k: int, seed: int = 0):
    first_index = int(np.random.default_rng(seed).integers(len(features)))
    representatives = farthest_first(features, k, first_index)
    return assign_to_centers(features, representatives), representatives


# --------------------------------------------------------------------------- #
# K-means
# --------------------------------------------------------------------------- #
def run_k_means(features: np.ndarray, k: int, seed: int = 0):
    """K-means with the nearest real member as cluster representative."""
    estimator = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = estimator.fit_predict(features)
    representatives = np.full(k, -1, dtype=int)
    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id)
        if len(members):
            distances = np.linalg.norm(
                features[members] - estimator.cluster_centers_[cluster_id], axis=1
            )
            representatives[cluster_id] = int(members[np.argmin(distances)])
    # Empty clusters cannot occur with n_init=10 on distinct points, but keep a
    # deterministic fallback so the function is total.
    for cluster_id in np.flatnonzero(representatives < 0):
        taken = representatives[representatives >= 0]
        candidates = np.setdiff1d(np.arange(len(features)), taken)
        distances = np.linalg.norm(
            features[candidates] - estimator.cluster_centers_[cluster_id], axis=1
        )
        representatives[cluster_id] = int(candidates[np.argmin(distances)])
    return labels.astype(int), representatives


# --------------------------------------------------------------------------- #
# K-medoids (FastPAM1)
# --------------------------------------------------------------------------- #
def initialize_medoids(distances: np.ndarray, k: int) -> np.ndarray:
    """PAM BUILD greedy initialisation (kept for API completeness)."""
    sample_indices = np.arange(len(distances))
    first_medoid = int(distances.sum(axis=1).argmin())
    medoids = [first_medoid]
    nearest_distances = distances[:, first_medoid]
    while len(medoids) < k:
        candidates = np.setdiff1d(sample_indices, medoids, assume_unique=True)
        candidate_distances = distances[:, candidates]
        improvements = (
            nearest_distances[:, None]
            - np.minimum(nearest_distances[:, None], candidate_distances)
        ).sum(axis=0)
        next_medoid = int(candidates[improvements.argmax()])
        medoids.append(next_medoid)
        nearest_distances = np.minimum(nearest_distances, distances[:, next_medoid])
    return np.asarray(medoids, dtype=int)


def _fit_k_medoids_once(distances: np.ndarray, medoids: np.ndarray, max_iter: int):
    """FastPAM1 best-improving swaps from one initialisation."""
    medoids = np.asarray(medoids, dtype=int).copy()
    sample_indices = np.arange(len(distances))

    for _ in range(max_iter):
        medoid_distances = distances[:, medoids]
        nearest_positions = medoid_distances.argmin(axis=1)
        nearest_distances = medoid_distances[sample_indices, nearest_positions]
        if len(medoids) == 1:
            second_nearest_distances = np.full(len(distances), math.inf)
        else:
            second_nearest_distances = np.partition(medoid_distances, 1, axis=1)[:, 1]

        best_delta = 0.0
        best_swap = None
        non_medoids = np.setdiff1d(sample_indices, medoids, assume_unique=True)
        for candidate in non_medoids:
            candidate_distances = distances[:, candidate]
            retained_distances = np.minimum(nearest_distances, candidate_distances)
            base_delta = float((retained_distances - nearest_distances).sum())
            replacement_delta = (
                np.minimum(second_nearest_distances, candidate_distances)
                - retained_distances
            )
            swap_deltas = base_delta + np.bincount(
                nearest_positions,
                weights=replacement_delta,
                minlength=len(medoids),
            )
            medoid_position = int(swap_deltas.argmin())
            if swap_deltas[medoid_position] < best_delta - 1e-12:
                best_delta = float(swap_deltas[medoid_position])
                best_swap = medoid_position, int(candidate)

        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]

    labels = distances[:, medoids].argmin(axis=1).astype(int)
    objective = float(distances[sample_indices, medoids[labels]].sum())
    return labels, medoids, objective


def _refine_two_swaps(distances: np.ndarray, medoids: np.ndarray, max_iter: int = 1):
    """Optional exhaustive 2-swap refinement for very small categories."""
    medoids = np.asarray(medoids, dtype=int).copy()
    sample_indices = np.arange(len(distances))
    for _ in range(max_iter):
        non_medoids = np.setdiff1d(sample_indices, medoids, assume_unique=True)
        if len(medoids) < 2 or len(non_medoids) < 2:
            break
        current_objective = float(distances[:, medoids].min(axis=1).sum())
        best_objective = current_objective
        best_swap = None
        for medoid_positions in itertools.combinations(range(len(medoids)), 2):
            kept_positions = [
                position
                for position in range(len(medoids))
                if position not in medoid_positions
            ]
            for candidates in itertools.combinations(non_medoids, 2):
                replacement = np.concatenate((medoids[kept_positions], np.asarray(candidates)))
                objective = float(distances[:, replacement].min(axis=1).sum())
                if objective < best_objective - 1e-12:
                    best_objective = objective
                    best_swap = medoid_positions, candidates
        if best_swap is None:
            break
        positions, candidates = best_swap
        medoids[list(positions)] = candidates
    return medoids


def fit_k_medoids(
    distances: np.ndarray,
    k: int,
    max_iter: int = 300,
    seed: int = 0,
    n_init: int = 10,
    two_swap_iter: int = 0,
):
    """Seeded multi-start FastPAM1 K-medoids.

    Every restart starts from a random medoid set drawn with *seed*; the
    restart with the lowest PAM objective is returned.  FastPAM1 computes every
    swap cost from each sample's nearest and second-nearest medoid distances, so
    it reaches the same optimum as PAM at a fraction of the cost.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not 1 <= k <= len(distances):
        raise ValueError(f"k must be in [1, {len(distances)}], got {k}")
    if max_iter <= 0 or n_init <= 0 or two_swap_iter < 0:
        raise ValueError("max_iter and n_init must be positive; two_swap_iter >= 0")

    rng = np.random.default_rng(seed)
    best_result = None
    for _ in range(n_init):
        initial_medoids = rng.choice(len(distances), size=k, replace=False)
        result = _fit_k_medoids_once(distances, initial_medoids, max_iter)
        for _ in range(two_swap_iter):
            refined = _refine_two_swaps(distances, result[1], max_iter=1)
            if np.array_equal(refined, result[1]):
                break
            result = _fit_k_medoids_once(distances, refined, max_iter)
        if best_result is None or result[2] < best_result[2] - 1e-12:
            best_result = result
    return best_result[0], best_result[1]


def run_k_medoids(features: np.ndarray, k: int, seed: int = 0, n_init: int = 10):
    distances = pairwise_distances(features, metric="euclidean")
    return fit_k_medoids(distances, k, seed=seed, n_init=n_init)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def cluster(
    features: np.ndarray,
    method: str,
    k: int,
    seed: int = 0,
    coreset_dim: int = 128,
):
    """Run *method* and return ``(labels, representative_indices)``.

    ``labels`` is ``None`` for the two pure samplers (``coreset``/``random``),
    which do not partition the training set.
    """
    if method == "coreset":
        return None, run_coreset(features, k, seed, coreset_dim)
    if method == "k_center":
        return run_k_center(features, k, seed)
    if method == "k_means":
        return run_k_means(features, k, seed)
    if method == "k_medoids":
        return run_k_medoids(features, k, seed)
    raise ValueError(f"unsupported clustering method: {method}")


def choose_k(
    features: np.ndarray,
    method: str,
    max_k: int,
    seed: int = 0,
    coreset_dim: int = 128,
) -> int:
    """Select k automatically by silhouette score (``--k 0``)."""
    from sklearn.metrics import silhouette_score

    sample_count = len(features)
    if sample_count <= 2:
        return sample_count
    upper = min(max_k, sample_count - 1)
    if upper < 2:
        return 1
    best_k, best_score = 2, -math.inf
    for candidate in range(2, upper + 1):
        labels, _ = cluster(features, method, candidate, seed, coreset_dim)
        if labels is None:
            raise ValueError("automatic k is only available for clustering methods")
        unique = np.unique(labels)
        if len(unique) < 2 or len(unique) >= sample_count:
            continue
        score = silhouette_score(features, labels, metric="euclidean")
        if score > best_score:
            best_k, best_score = candidate, float(score)
    print(f"auto-selected k={best_k} (silhouette={best_score:.6f})")
    return best_k
