# Results

Every number reported in the paper and in the supplementary material is
regenerated here from raw files that are **also** shipped in this directory.
Nothing is hand-edited.

```
results/
├── MANIFEST.csv                  source, row count and SHA-256 of every raw file
├── one_shot/                     complete per-reference 1-shot sweeps (5 methods)
│   └── <method>/{mvtec,visa}.csv
├── centers_raw/                  raw per-centre evaluations (8 methods × 62 datasets)
│   └── <Method>/<shot>-shot/center(s)_<ids>_<dataset>.csv
├── validation/                   Table I (Sec. III-D)
└── reevaluation/                 Table II, Fig. 5 (Sec. IV)
```

## `one_shot/` — the complete 1-shot table (12 288 rows per method)

One row per normal training image used as the single reference, for the five
methods whose `μ_g` is affordable.  These files are the only inputs of Table I:
all six strategies read different rows of the same table.

| Method | Rows (MVTec + VisA) | Columns |
|---|---|---|
| `patchcore` | 3 629 + 8 659 | `dataset, object, Run, ref_sample, reference_index, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO, elapsed_seconds` |
| `winclip` | 3 629 + 8 659 | `object, ref_sample, reference_index, Image AUROC, Image AUPR, Pixel AUPRO, Pixel AUROC` |
| `anomalydino` | 3 629 + 8 659 | `object, Total_Train_Images, ref_index, ref_sample, run_dir, Image AUROC, Image AUPR, classification_F1, Pixel AUPRO, Pixel AUROC, seg_F1, mean_inference_time, status` |
| `inpformer` | 3 629 + 8 659 (split by the `dataset` column) | `dataset, object, Run, ref_sample, 参考索引, ref_path, Image AUROC, Image AUPR, Image F1, Pixel AUROC, Pixel AUPR, Pixel F1, Pixel AUPRO, …` |
| `subspacead` | 3 629 + 8 659 | `object, Run, ref_sample, 参考索引, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO` |

`goef-validate` only needs `object`, `ref_sample` and the four metric columns;
the extra columns are the methods' own bookkeeping and are kept for provenance.

## `centers_raw/` — raw per-centre evaluations of the eight methods

496 files = 8 methods × 31 centre datasets × 2 benchmarks, at all five shots
(`5 + 10 + 10 + 5 + 1` centre combinations).  Each file uses the unified
11-column schema:

```
Category,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO,Images,Inference Total (s),Inference (ms/image),Throughput (images/s),GPU Peak Allocated (MiB),GPU Peak Reserved (MiB)
```

PatchCore additionally writes a native `results.csv`; only its `*_unified.csv`
is collected.  PromptAD's `_cls.csv` / `_seg.csv` training logs are not
collected (they are not comparable results).  Every file is listed in
[`MANIFEST.csv`](MANIFEST.csv) with its source path and SHA-256.

## `validation/` — Table I (Sec. III-D)

| File | Rows | Contents |
|---|---|---|
| `table_validation_paper_exact.csv` | 30 | the published table: `ϵ̄ ×100` and `r̄` per strategy, `Metric=ALL` plus one scope per metric |
| `table_validation.csv` | 30 | the same computation with a uniform six-decimal rule (differs only in `SINGLE` and one thousandth of `RE-CS`) |
| `validation_details.csv` | 16 200 | one row per `(dataset, detector, seed, strategy, category, metric)` with `estimate`, `actual`, `abs_error` and `rank` |

Comparisons: 2 datasets × 5 detectors × 5 seeds × 27 categories × 4 metrics =
2 700 per strategy, 16 200 in total.

## `reevaluation/` — Table II and Fig. 5 (Sec. IV)

| File | Rows | Contents |
|---|---|---|
| `all_final_weighted_results.csv` | 80 | Table II: 5 shots × 2 datasets × 8 methods × 4 metrics |
| `category_weighted_results.csv` | 1 080 | the per-category weighted metrics `M(c)` of Eq. (21) that are averaged into Table II |
| `final_weighted_results.csv` | 120 | the 80 dataset rows plus 40 `overall` rows (equal-weight MVTec-AD/VisA mean) |
| `method_metric_rankings.csv` | 480 | `Shot,Metric,Scope,Rank,Method,Score` |
| `metric_rank_frequency.csv` | 80 | rank 1…8 counts and the average rank per method and table |
| `figure5_ranking.csv` | 8 | average performance metric and average rank over the 1/2/4-shot tables (Fig. 5) |

Shots 1–5 are computed; the paper reports 1, 2 and 4.

## Verification

| Check | Result |
|---|---|
| Table II vs the archived summary (80 rows) | identical (≤ `5e-7` before rounding) |
| Table II rankings (480 rows) | identical |
| Rank frequencies | identical |
| Fig. 5 averages vs the paper / supplementary Fig. 1 | identical to 2 decimals |
| Table I `ϵ̄` and `r̄` (paper-exact mode) | identical to 3 decimals |
| Weights vs the paper's per-shot weight tables (k = 1…5) | max deviation `5e-13` |

## Regenerating

```bash
./scripts/06_aggregate_validation.sh     # results/validation/
./scripts/07_aggregate_reevaluation.sh   # results/reevaluation/
./scripts/08_collect_results.sh          # results/one_shot/, results/centers_raw/
```

The aggregation stages only read CSVs and run on CPU in a few seconds; the
collector only copies files and writes the manifest.
