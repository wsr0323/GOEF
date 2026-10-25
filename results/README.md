# Regenerated paper artefacts

Everything here is produced by the code in this repository from the eight
methods' per-image / per-centre result CSVs.  Nothing is hand-edited.

## `validation/` — Sec. III-D, Table I

| File | Rows | Contents |
|---|---|---|
| `table_validation.csv` | 30 | `Metric,Strategy,epsilon_bar_x100,epsilon_bar_pooled_x100,rank_bar,experiments`; `Metric=ALL` plus one scope per metric. Six-decimal rule applied to every strategy. |
| `table_validation_paper_exact.csv` | 30 | Same table with the archived pipeline's precision asymmetry for `SINGLE`, i.e. the published numbers. |
| `validation_details.csv` | 16 200 | One row per `(dataset, detector, seed, strategy, category, metric)` with `estimate`, `actual`, `abs_error` and `rank`. |

Comparisons: 2 datasets × 5 detectors × 5 seeds × 27 categories × 4 metrics =
2 700 per strategy, 16 200 in total.

## `reevaluation/` — Sec. IV, Table II

| File | Rows | Contents |
|---|---|---|
| `all_final_weighted_results.csv` | 80 | `Shot,Dataset,Method,Category Count,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO` — the paper's Table II |
| `final_weighted_results.csv` | 120 | the 80 dataset rows plus 40 `overall` rows (equal-weight MVTec-AD/VisA mean) |
| `method_metric_rankings.csv` | 480 | `Shot,Metric,Scope,Rank,Method,Score`; scopes `mvtec`, `visa`, `overall` |
| `metric_rank_frequency.csv` | 80 | rank 1…8 counts and the average rank of every method (Fig. 5) |

Shots 1–5 are included; the paper reports 1, 2 and 4.

## Verification

| Check | Result |
|---|---|
| Table II vs the archived summary (80 rows) | max deviation `5e-7` |
| Table II rankings (480 rows) | identical |
| Rank frequencies | identical |
| Table I `ϵ̄` (six strategies) | identical to 3 decimals |
| Table I `r̄` | identical, except `SINGLE` `4.473` vs `4.48` for `table_validation.csv` (see `docs/results.md`) |

## Regenerating

```bash
./scripts/06_aggregate_validation.sh     # results/validation/
./scripts/07_aggregate_reevaluation.sh   # results/reevaluation/
```

Both stages only read CSVs, so they run on CPU in a few seconds.
