# Reproduced results

Both tables of the paper are regenerated from the artefacts in `results/`.
`results/validation/` is produced by `scripts/06_aggregate_validation.sh`,
`results/reevaluation/` by `scripts/07_aggregate_reevaluation.sh`.

## Table I — validation of the six evaluation strategies

`ϵ̄` is the average approximation error `|μ_g − μ_r|` multiplied by 100, `r̄` the
average rank of `ε`.  Every strategy is evaluated on 2 datasets × 5 detectors ×
5 seeds × 27 categories × 4 metrics = **2 700 comparisons**.

| Metric | SINGLE | RE-RS | RE-CS | GOEF-KC | GOEF-KD | GOEF-KM |
|---|---|---|---|---|---|---|
| `ϵ̄ (×100)` | 1.631 | 0.693 | 0.809 | 0.733 | 0.551 | **0.489** |
| `r̄` | 4.49 | 3.16 | 3.53 | 3.44 | 3.04 | **2.93** |

This is exactly the paper's table (file
`table_validation_paper_exact.csv`).  Source values:

| Strategy | `ϵ̄ ×100` | `r̄` |
|---|---|---|
| GOEF-KM | 0.488820 | 2.927037 |
| GOEF-KD | 0.551166 | 3.039259 |
| RE-RS | 0.693031 | 3.157407 |
| GOEF-KC | 0.733179 | 3.444815 |
| RE-CS | 0.808632 | 3.525556 |
| SINGLE | 1.631410 | 4.494444 |

Per-metric breakdown (`table_validation.csv`, `Metric != ALL`):

| Metric | SINGLE | RE-RS | RE-CS | GOEF-KC | GOEF-KD | GOEF-KM |
|---|---|---|---|---|---|---|
| Image AUROC | 2.633 | 1.096 | 1.168 | 1.049 | 0.800 | **0.730** |
| Image AUPR | 1.874 | 0.737 | 0.824 | 0.834 | 0.579 | **0.501** |
| Pixel AUROC | 0.437 | 0.226 | 0.312 | 0.225 | 0.194 | **0.160** |
| Pixel AUPRO | 1.363 | 0.622 | 0.802 | 0.707 | 0.527 | **0.486** |

GOEF-KM wins on every metric and GOEF-KD is second on three of the four.
Coreset and k-center are consistently worse than random sampling, and k-center
is the worst of the three clustering strategies: the outliers they select have
small cluster weights but atypical performance, exactly as predicted by the
analysis of Sec. III-C.  Pixel AUROC is the one metric where the ordering of
RE-RS and GOEF-KC almost swaps (0.2264 vs 0.2249), i.e. the metric on which the
clustering advantage is smallest.

### The two Table I files

| File | What it is |
|---|---|
| `table_validation_paper_exact.csv` | reproduces the published numbers, including the way the archived pipeline handled SINGLE |
| `table_validation.csv` | the same computation with a single, uniform precision rule |

Metrics are stored with six decimals and compared with exact decimal
arithmetic, exactly as in the paper's pipeline (`goef.validation` uses
`decimal.Decimal` on the six-decimal strings, so ties are detected precisely).

The archived pipeline copied the SINGLE baseline from the raw metric CSV at
full precision while storing the other five strategies with six decimals.
Because `r̄` is a competition rank, that asymmetry slightly *disadvantages*
SINGLE (4.4944 instead of 4.4804).  `goef-validate --legacy_single_precision`
reproduces it; the default applies the same six-decimal rule to all six
strategies.  The difference is confined to `SINGLE` (0.014) and `RE-CS`
(0.0007); every `ϵ̄` value and every other rank is identical to the paper.

## Table II — re-evaluation of eight FS-IAD methods

Weighted with Eq. (21)-(22) over the 5-cluster GOEF partition.  All numbers are
percentages and match the paper's Table II to the last reported digit.

### 1-shot

| Method | MVTec I-AUROC | I-AUPR | P-AUROC | P-AUPRO | VisA I-AUROC | I-AUPR | P-AUROC | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| PatchCore | 83.64 | 92.41 | 91.70 | 77.94 | 77.07 | 80.42 | 93.11 | 71.89 |
| WinCLIP | 93.42 | 96.78 | 93.55 | 84.22 | 81.63 | 83.05 | 94.89 | 80.84 |
| PromptAD | 93.57 | 96.96 | 95.50 | 88.85 | 86.82 | 89.08 | 96.49 | 82.35 |
| AnomalyDINO | **96.42** | 98.04 | 96.85 | 92.96 | 89.60 | 90.60 | 97.74 | 93.30 |
| FoundAD | 94.68 | 97.22 | 96.77 | 93.14 | 92.01 | 90.88 | 98.25 | 93.39 |
| INP-Former | 95.43 | 97.57 | 96.81 | 92.42 | 93.08 | 93.32 | 97.04 | 92.33 |
| UniVAD | 96.29 | **98.21** | 96.23 | 92.78 | **93.62** | **94.05** | 97.97 | 93.27 |
| SubspaceAD | 95.68 | 97.75 | **97.50** | **94.38** | 91.99 | 91.95 | **98.30** | **94.73** |

### 2-shot

| Method | MVTec I-AUROC | I-AUPR | P-AUROC | P-AUPRO | VisA I-AUROC | I-AUPR | P-AUROC | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| PatchCore | 86.46 | 93.69 | 93.00 | 80.75 | 80.45 | 82.90 | 94.46 | 75.57 |
| WinCLIP | 94.30 | 97.15 | 93.90 | 84.96 | 83.19 | 84.50 | 95.11 | 81.24 |
| PromptAD | 94.55 | 97.46 | 95.86 | 89.52 | 87.91 | 90.04 | 96.80 | 82.85 |
| AnomalyDINO | 97.16 | 98.40 | 97.11 | 93.39 | 91.22 | 92.00 | 97.94 | 93.92 |
| FoundAD | 95.94 | 97.85 | 97.01 | 93.64 | 93.35 | 92.19 | 98.40 | 93.99 |
| INP-Former | 96.60 | 98.24 | 97.13 | 92.98 | 94.62 | 94.41 | 97.36 | 93.26 |
| UniVAD | **97.39** | **98.73** | 96.46 | 93.15 | **94.76** | **95.06** | 98.21 | 94.03 |
| SubspaceAD | 96.57 | 98.13 | **97.71** | **94.69** | 92.78 | 92.59 | **98.42** | **95.11** |

### 4-shot

| Method | MVTec I-AUROC | I-AUPR | P-AUROC | P-AUPRO | VisA I-AUROC | I-AUPR | P-AUROC | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| PatchCore | 89.50 | 95.16 | 94.18 | 83.31 | 83.12 | 85.13 | 95.28 | 78.17 |
| WinCLIP | 95.07 | 97.48 | 94.21 | 85.66 | 84.61 | 85.94 | 95.28 | 81.52 |
| PromptAD | 95.38 | 97.90 | 96.19 | 90.15 | 88.57 | 90.61 | 97.05 | 83.31 |
| AnomalyDINO | 97.74 | 98.70 | 97.31 | 93.71 | 92.42 | 93.07 | 98.09 | 94.40 |
| FoundAD | 96.69 | 98.26 | 97.16 | 93.88 | 93.89 | 92.82 | 98.47 | 94.28 |
| INP-Former | 97.51 | 98.72 | 97.37 | 93.37 | **95.62** | **95.15** | 97.60 | 93.96 |
| UniVAD | **98.27** | **99.15** | 96.64 | 93.44 | 95.38 | 95.55 | 98.33 | 94.47 |
| SubspaceAD | 97.46 | 98.54 | **97.87** | **94.94** | 93.35 | 93.13 | **98.50** | **95.35** |

## Fig. 5 — average performance vs average rank

`results/reevaluation/figure5_ranking.csv` averages the four metrics over the
three reported settings (1-, 2- and 4-shot) and both datasets, and averages each
method's tied rank over the six resulting tables:

| Method | Average performance metric | Average rank number |
|---|---|---|
| SubspaceAD | 95.73 | **2.29** |
| UniVAD | **95.89** | 2.42 |
| AnomalyDINO | 95.09 | 3.33 |
| INP-Former | 95.50 | 3.42 |
| FoundAD | 95.17 | 3.54 |
| PromptAD | 91.82 | 6.00 |
| WinCLIP | 89.27 | 7.04 |
| PatchCore | 85.80 | 7.96 |

This reproduces both the paper's Fig. 5 and the supplementary Fig. 1, including
the two headline claims: UniVAD has the highest average performance metric
(95.89 %) and SubspaceAD the best average rank (2.29).

`results/reevaluation/metric_rank_frequency.csv` holds the underlying per-table
counts (rank 1…8 per method and their average rank);
`method_metric_rankings.csv` holds the 480 individual rankings
(5 shots × 4 metrics × 3 scopes × 8 methods).

## Reproducing

```bash
export GOEF_WORKSPACE=/data/goef-work GOEF_DATA=/data/datasets
export GOEF_DINOV3=/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
./scripts/run_all.sh

# reproduce the published Table I row of SINGLE as well
python -m goef.cli.validate --config "$GOEF_WORKSPACE/outputs/validation/validation_config.yaml" \
  --legacy_single_precision --output /tmp/validation_paper_exact
```

## Verification performed for this release

| Check | Result |
|---|---|
| Eq. (22) weights vs the per-shot weight tables `k = 1…5` | max deviation `5e-13` |
| Table II vs the archived `all_final_weighted_results.csv` (80 rows) | identical (the archived names `patchcore` / `Accurate-WinCLIP` map to the paper's `PatchCore` / `WinCLIP`) |
| Table II rankings vs archived `method_metric_rankings.csv` (480 rows) | identical |
| Rank frequencies vs archived `metric_rank_frequency_summary.csv` | identical |
| Fig. 5 averages vs the paper and the supplementary Fig. 1 | identical to 2 decimals |
| Table I `ϵ̄` and `r̄` (paper-exact mode) | identical to 3 decimals |
| Table I from the collected `results/one_shot/` | reproduces the same table as the original sources |
| Table II from the collected `results/centers_raw/` | reproduces the same table as the original sources |
