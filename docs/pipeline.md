# Pipeline

Seven stages, each idempotent and restartable.  All paths are taken from
`scripts/env.sh`.

```
datasets ─► [1] selection ─► [2] centres ─► [5] method runs ─► [7] aggregation ─► Table II
                  │                                                  ▲
                  ├─► [3] weights ──────────────────────────────────┘
                  │
                  └─► [4] per-reference sweeps ─► [6] aggregation ─► Table I
```

## Stage 1 — representative selection (`goef-sample`)

Runs the five selection strategies on every category of MVTec-AD and VisA for
every seed.

```
sampling/<method>/<Dataset>_k5_seed<seed>/
├── run_config.json
├── summary.csv                  method,category,cluster_id,count,representative_image
└── <category>/
    ├── selected_samples.csv     sample_id,filename,full_path
    ├── representative_mapping.csv
    ├── clusters.csv             path,filename,cluster        (clustering methods)
    └── clusters_preview.jpg     (only with --preview)
```

`count` is the cluster size `N_j`; `representative_image` is the file name of
`Ī_j` inside `<category>/train/good`.  For `random` and `coreset` every
`count` is 1 because those strategies sample `n` independent references rather
than partitioning the space.

`random` is additionally run with `k=1` (`random/<Dataset>_k1_seed<seed>`) to
provide the **SINGLE** baseline of Table I.

## Stage 2 — centre materialisation (`goef-build-centers`)

Builds one dataset per centre combination from the `k_means`, `k=5`, `seed=1234`
partition (the GOEF default after the empirical study).

```
centers/<shot>-shot/{mvtec,visa}/
├── center_01/            (1-shot, 5 datasets)
│   ├── meta.json         {"train": {cat: [k entries]}, "test": {cat: [...]}}
│   ├── representatives.csv
│   ├── dataset_info.json
│   └── <category>/{train/good,test,ground_truth}
└── centers_01_02/ ...    (2-shot, 10 datasets)
```

`--materialize` controls whether images are copied, symlinked, hard-linked or
only described in `meta.json`.  `symlink` (default) keeps the tree cheap;
`copy` is required by methods that re-scan the directory.

The test split and every ground-truth mask are preserved bit-for-bit, so a
centre dataset is a drop-in replacement for the original benchmark.

## Stage 3 — weights (`goef-weights`)

```
sampling/weights/mvtec_k2.csv
dataset,category,shot,evaluation_shot,centers,count,total_count,weight
mvtec,bottle,2,1,01,1431,21736,0.065835480309
mvtec,bottle,2,2,01_02,1998,21736,0.091921236658
...
```

* `shot` — the evaluated shot (`k`);
* `evaluation_shot` — the shot of the dataset that must be run, `q = |τ| ≤ k`;
* `centers` — zero-padded cluster ids of the unique cluster set;
* `count` — `N_τ` (Eq. 22), `total_count` — `C(N, k)`;
* `weight` — `N_τ / C(N,k)`; the weights of a category sum to 1.

Weights are also written per shot (`mvtec_k1.csv` … `mvtec_k5.csv`) and for both
datasets together (`mvtec_all.csv`).

## Stage 4 — per-reference sweeps (Table I)

For the five methods whose `μ_g` is affordable, every training image is used as
the single 1-shot reference:

```
results/<detector>/{mvtec,visa}.csv
object,ref_sample,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO   (+ extra columns)
```

Contract: one row per `(category, reference image)`.  Additional columns such as
`Run`, `reference_index`, `参考索引`, timings or GPU peaks are ignored.

## Stage 5 — per-centre evaluations (Table II)

Every method is run on every centre dataset:

```
third_party/<Project>/results/<shot>-shot/<center>_<dataset>.csv
Category,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO,Images,Inference Total (s),...
bottle,0.9968253968253968,0.9989838017340629,0.9888589539541247,0.9645315293316139,83,...
...,Overall,...
```

62 files per method (31 centre directories × 2 datasets) for each shot, i.e.
`5 + 10 + 10 + 5 + 1 = 31` centres.

## Stage 6 — validation aggregation (Table I)

`goef-validate` reads the `summary.csv` of every strategy and the per-reference
CSVs, then:

1. computes `μ_g` per category (`goef.validation.actual_metrics`);
2. computes `μ_r` per strategy (`weighted_metric` over `N_j/N`);
3. computes `ε = |μ_g − μ_r|` at six-decimal precision;
4. ranks the six strategies per (dataset, detector, seed, category, metric);
5. averages `ε` per table and over the 50 tables → `ϵ̄`; averages ranks → `r̄`.

```
results/validation/
├── table_validation.csv      Metric,Strategy,epsilon_bar_x100,epsilon_bar_pooled_x100,rank_bar,experiments
└── validation_details.csv    one row per comparison (16 200 rows)
```

## Stage 7 — re-evaluation aggregation (Table II)

`goef-reevaluate` joins the weights of stage 3 with the results of stage 5:

1. for each category, `M(c) = Σ_τ w_τ M_q(c; τ)` (Eq. 21);
2. dataset value = macro average over categories; overall = mean of MVTec-AD and VisA;
3. ranking per shot, metric and scope; rank frequencies for Fig. 5.

```
results/reevaluation/
├── final_weighted_results.csv       80 dataset rows + 40 overall rows
├── all_final_weighted_results.csv   80 rows (Table II)
├── method_metric_rankings.csv       480 rows (5 shots × 4 metrics × 3 scopes × 8 methods)
└── metric_rank_frequency.csv        80 rows (+ per-shot average ranks)
```

Shots with an incomplete grid are skipped and reported, never re-normalised —
the same rule as the paper's pipeline.

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| `missing ...-shot/01` | the method did not run on that centre dataset |
| `no representative for <category>` | the selection was run with a category filter |
| weights sum ≠ 1 | `summary.csv` was edited or truncated |
| centre build fails on `representative differs from source` | the selected image was modified after selection (use `--skip-representative-content-check` only for debugging) |
