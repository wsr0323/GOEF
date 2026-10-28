# Methodology → code map

This page links every equation and algorithm of the paper to the code that
implements it.  Symbol names follow the paper.

## 1. The ideal globality-preserving metric

| Paper | Meaning | Code |
|---|---|---|
| Eq. (1) | `μ_g = 1/N Σ_i P1(I_i)`: mean over *all* normal images | `goef.validation.actual_metrics` |
| Eq. (2) | `μ_r = Σ_j N_j P1(Ī_j) / N` | `goef.aggregation.weighted_metric` with `RunSelection.weights()` |
| Eq. (3) | `ε = |μ_g − μ_r|` | `goef.validation.validate_dataset` (`abs_error`) |
| Eq. (4)-(5) | Lipschitz bound `ε ≤ L/N Σ_j Σ_m ||φ(Ī_j) − φ(I_m)||` | motivates the clustering objectives below |
| Eq. (6) | K-means objective, i.e. the bound of Eq. (5) | `goef.selection.run_k_means` |
| Eq. (7) | K-medoids objective | `goef.selection.fit_k_medoids` |

`μ_g` is only computable when *every* training image can be used as a reference,
which is what the per-reference sweeps of `modifications/<method>/validation/` do.  For a
category with `N` training images and a test set of `T` images this costs `N`
full evaluations, so `μ_g` is affordable for 1-shot only.

## 2. Algorithm 1 — GOEF for 1-shot

```
1  E ← {φ(I_i)}                       goef.features.extract_embeddings
2  {C_j} ← S(E, n)                    goef.selection.cluster
4  Ī_j ← Rep_S(C_j)                   run_k_means / run_k_medoids / run_coreset / random
5  N_j ← |C_j|                        summary.csv: count
7  μ_r ← Σ_j N_j P1(Ī_j) / N          goef.aggregation.weighted_metric
```

`python -m goef.cli.sample --method k_means --k 5 --seed 0 …` performs steps 1-5
and writes `summary.csv` (the `count` column is `N_j`, `representative_image` is
`Ī_j`).  `python -m goef.cli.validate` performs step 7 for all six strategies and
computes `ε`.

Note the "small trick" of Sec. III-C: K-means centroids are not images, so the
member closest to the centroid is returned (`run_k_means`).  K-medoids already
returns real images (`run_k_medoids`).  K-center is included as the negative
control: it minimises the maximum radius and therefore prefers outliers
(`run_k_center`).

## 3. Eq. (15)-(20) — from 1-shot to k-shot

The paper proves that `P_k` satisfies the same Lipschitz condition with
`L_k = max(L_1, L_{k-1})`, so the bound of `ε` is again a sum of distances
between training images and their representatives.  The same clustering
therefore lowers the error for every `k`.

## 4. Eq. (21)-(22) and Algorithm 2 — efficient k-shot weights

The naive sum of Eq. (17)-(19) runs over `N^(k)` ordered k-tuples.  Two
relaxations make it tractable:

1. **order invariance** — `P_k(I_{j1}, …, I_{jk}) = P_k(I_{j1'}, …, I_{jk'})` for
   any permutation, because mainstream detectors are order-insensitive;
2. **repetition collapse** — `P_k(Ī_1, Ī_1, Ī_2, Ī_2) = P_2(Ī_1, Ī_2)`, because
   repeating a representative adds no information.

Consequently every k-tuple is represented by its unique cluster set `τ`
(`q = |τ| ≤ k`), which is evaluated once as a **q-shot** experiment.  With
`a_s` images drawn from cluster `τ_s`:

```
N_τ = Σ_{a_1+…+a_q=k} Π_s C(N_{τ_s}, a_s)          Eq. (22) after cancelling k!
w_τ = N_τ / C(N, k)
```

| Paper | Code |
|---|---|
| Eq. (21) | `goef.reevaluate.aggregate_method` |
| Eq. (22) | `goef.weights.shot_weights` |
| Algorithm 2, line 7 (`N^(k)`) | `goef.weights.total_draws` |
| Algorithm 2, lines 9-18 | `goef.weights.category_weights` |
| Algorithm 2, lines 12-15 (reuse lower shots) | `WeightRow.evaluation_shot = q` |
| Algorithm 2, line 17 | `goef.aggregation.weighted_metric` |

`goef.weights` is the only place where the combinational arithmetic lives; it is
validated against the per-shot weight tables of the paper's own pipeline for
`k = 1…5` (max deviation `5e-13`, see `tests/test_weights.py`).

### Worked example (MVTec-AD `bottle`, `N = 209`, clusters `54/37/57/24/37`)

| shot | centre set | `N_τ` | `w_τ` |
|---|---|---|---|
| 1 | `01` | 54 | 54/209 = 0.2584 |
| 2 | `01_02` | 54·37 = 1998 | 1998/C(209,2) = 0.0919 |
| 4 | `01_02` | C(54,3)·37 + C(54,2)·C(37,2) + 54·C(37,3) = 2 290 374 | 0.0297 |
| 4 | `01_02_03_04` | … | see `results` |

The third row is the reuse mechanism: three 4-tuples `(1,1,1,2)`, `(1,1,2,2)`,
`(1,2,2,2)` share the unique set `{1,2}` and are therefore evaluated by the
**2-shot** centre dataset `centers_01_02`, with the summed weight of all three.

## 5. The four metrics

| Metric | Implementation |
|---|---|
| Image AUROC | `sklearn.metrics.roc_auc_score` on image scores |
| Image AUPR | `sklearn.metrics.average_precision_score` |
| Pixel AUROC | `roc_auc_score` over all pixels of a category |
| Pixel AUPRO | `unified_evaluation.compute_aupro`: 8-connected ground-truth regions, 200 thresholds uniform in background FPR, integrated on `[0, 0.30]`, divided by `0.30` |

`modifications/common/unified_evaluation.py` is copied unchanged into every project
so that no adapter can silently change the protocol.  The per-method AUPRO
implementations of the upstream projects are **not** used for the paper's
numbers: every adapter reports the four metrics above.

## 6. Why GOEF changes conclusions

Two consequences of Eq. (2) and Eq. (21) are visible in the results:

* a strategy that covers the normal space with compact clusters (`k-means`,
  `k-medoids`) is a better estimator of `μ_g` than random or coreset sampling;
* a strategy that maximises coverage (`k-center`, coreset) systematically
  selects outliers and is *worse* than random sampling — its `N_j` weights are
  small but its representatives are atypical.
