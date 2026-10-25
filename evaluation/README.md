# Method adapters

This directory contains the **only** code that differs between the eight
FS-IAD methods re-evaluated in the paper. Every adapter is a thin driver: it
loads a center dataset produced by GOEF, runs the method, and writes a CSV with
the four shared metrics (`Image AUROC`, `Image AUPR`, `Pixel AUROC`,
`Pixel AUPRO`).

The repository does **not** vendor the upstream projects. `scripts/install_methods.sh`
clones each upstream at the pinned revision and copies the files below into it
(see `third_party/README.md` for provenance and licences).

```
evaluation/
├── common/unified_evaluation.py     # the single metric implementation (Sec. III-D1)
├── validation/                      # Sec. III-D: every training image as 1-shot
│   ├── patchcore/ winclip/ inpformer/ subspacead/ anomalydino/
└── reevaluation/                    # Sec. IV: the 62 GOEF center datasets
    ├── patchcore/ winclip/ anomalydino/ inpformer/
    └── subspacead/ promptad/ foundad/ univad/
```

## Two different experiments

| Experiment | Paper | Reference images | Adapter dir |
|---|---|---|---|
| Per-reference sweep | Sec. III-D (Table I) | **every** image in `train/good`, one run per image | `validation/` |
| Per-center evaluation | Sec. IV (Table II) | the k images of one GOEF center dataset | `reevaluation/` |

The per-reference sweeps are only needed for the five methods whose `mu_g` is
affordable (PatchCore, WinCLIP, AnomalyDINO, INP-Former, SubspaceAD). The
per-center evaluation covers all eight methods.

## Shared metric protocol

`common/unified_evaluation.py` is copied unchanged into every project so that no
adapter can silently change the comparison protocol:

* `Image AUROC` – `sklearn.metrics.roc_auc_score` on image scores;
* `Image AUPR`  – `sklearn.metrics.average_precision_score` (not trapezoidal);
* `Pixel AUROC` – global `roc_auc_score` over all pixels of a category;
* `Pixel AUPRO` – per-region overlap, 8-connected ground-truth regions,
  thresholds uniform in global background FPR, integrated over `[0, 0.30]` and
  divided by `0.30`.

Every adapter calls `compute_metrics` / `make_result_row` / `write_results_csv`
from that file. FoundAD and PromptAD keep extra GPU-peak columns; downstream
code selects columns **by name**, never by position.

## Sec. III-D: per-reference sweeps

Each adapter writes one row per `(category, reference image)` with at least:

```
object,ref_sample,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO
```

Additional columns (`Run`, `reference_index`, `参考索引`, timing, …) are ignored
by `goef-validate`.

| Method | Entry point | Output |
|---|---|---|
| PatchCore | `bin/run_one_shot_patchcore.py` | `<out>/<dataset>_one_shot.csv` |
| WinCLIP | `run_one_category.py` | `--output_csv` |
| INP-Former | `INP_Former_One_Image_Sweep.py` | `--output-csv` |
| SubspaceAD | `run_mvtec_each_train_1shot.sh`, `run_visa_each_train_1shot.sh` + `merge_shot_results.py` | `few_shot_results_each_train_1shot/*/all_runs_table.csv` |
| AnomalyDINO | `run_{mvtec,visa}_train_1shot_refs.py` | `<output_root>/summary.csv` |

`scripts/04_run_one_shot_sweeps.sh` invokes them with the paths used in the
paper. All five are resumable: they skip references already present in their
CSV, so an interrupted sweep can be restarted.

## Sec. IV: per-center evaluations

Every adapter consumes one center dataset
(`<center_root>/<shot>-shot/<center>/<dataset>`) and writes

```
<project>/results/<shot>-shot/<center>_<dataset>.csv
```

with the `unified_evaluation.RESULT_COLUMNS` schema, plus an `Overall` row.
PatchCore additionally writes the native `results.csv` schema and a
`*_unified.csv`; `goef-reevaluate` prefers the unified one.

| Method | Driver | Notes |
|---|---|---|
| PatchCore | `train_all_shots.sh [shots…]` | `GPU`, `SEED`, `SAVE_MODELS` |
| WinCLIP | `run_center_shots.sh` | `GPU_ID`, `MODEL`, `IMAGE_SIZE` |
| AnomalyDINO | `train_all_few_shot.sh` | `SKIP_EXISTING=1`, `RESOLUTION=672` |
| INP-Former | `train_all_few_shot.sh` | `--shot k`, 200 epochs |
| SubspaceAD | `scripts/benchmark_center_few_shot.sh [shots…]` | res 518, DINOv2-g |
| PromptAD | `run_center_shot_unified.py --shots …` | 100 epochs, 3 CSVs per task |
| FoundAD | `batch_few_shot.py --shots …` | trains + tests per center |
| UniVAD | `run_fewshot_centers.sh [shots…]` | needs `prepare_fewshot_masks.py` once |

All drivers accept a subset of shots, so a single setting can be reproduced
without recomputing the other four.

## Centre directory naming

GOEF writes `center_01` for 1-shot and `centers_01_02` for k ≥ 2. Some upstream
drivers normalise the name to `center_01_02` in their output CSV. Both spellings
are accepted by `goef-reevaluate` (`centers?_<ids>_<dataset>.csv`).
