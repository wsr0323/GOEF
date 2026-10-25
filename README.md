# GOEF — Globality-Orientated Evaluation Framework

Reference implementation of

> **Don't Miss Forest for A Tree: The Globality-Orientated Evaluation Framework for
> Few-shot Image Anomaly Detection**
> Siqi Wang, Shiru Wang, Rundong Xian, Guangpu Wang, Sha He, Chuanfu Xu, Jie Liu, Xinwang Liu

Few-shot image anomaly detection (FS-IAD) is usually evaluated with a single
manually or randomly picked normal reference image. That image is a **local**
point of the normal image space, so the reported score can be several AUROC
points away from the **global** performance, and different papers use different
reference images, which makes their numbers incomparable.

GOEF replaces the local reference with a *globality-preserving* estimate:

1. embed every normal training image with a frozen encoder (DINOv3);
2. cluster them into `n` compact clusters (K-means / K-medoids);
3. use one representative image per cluster as a 1-shot reference;
4. weight each representative by the size of its cluster — Eq. (2);
5. for `k > 1`, weight every k-shot centre combination by the number of
   k-subsets that map onto it — Eq. (21)–(22).

This repository contains the code of **every** stage used in the paper: the
representative selection, the k-shot centre materialisation, the weight
computation, the shared metric protocol, the empirical validation of Sec. III-D
and the eight-method re-evaluation of Sec. IV.

---

## What is in the box

| Path | Contents |
|---|---|
| `goef/` | the framework: embeddings, selectors, centres, weights, aggregation, validation |
| `goef/cli/` | five command-line entry points (`sample`, `build_centers`, `weights`, `validate`, `reevaluate`) |
| `evaluation/` | the per-method adapters for the 5 validation sweeps and the 8 re-evaluated methods |
| `scripts/` | the end-to-end pipeline (`run_all.sh` + seven numbered stages) |
| `configs/` | YAML configurations of the two aggregation stages |
| `results/` | the regenerated Table I and Table II artefacts |
| `docs/` | methodology → code map, dataset layout, reproduction notes, third-party provenance |
| `tests/` | unit tests for the weight and aggregation arithmetic |

```
                    normal training images
                             │
                    goef.features (DINOv3)
                             ▼
                    goef.selection (5 strategies)
                             │   summary.csv  (representative + cluster size)
              ┌──────────────┴───────────────┐
              ▼                              ▼
      goef.validation                 goef.centers + goef.weights
   (Sec. III-D, Table I)            (k-shot centre datasets, Eq. 21–22)
   μ_g vs μ_r, ε and rank                     │
              │                              ▼
              │                     evaluation/ (8 FS-IAD methods)
              │                              │
              │                     goef.reevaluate (Sec. IV, Table II)
              ▼                              ▼
        results/validation            results/reevaluation
```

## Installation

```bash
git clone <this-repo> GOEF && cd GOEF
python -m pip install -e .            # core framework (numpy, scikit-learn, pillow, pyyaml)
```

The method adapters need the upstream projects and their own environments:

```bash
export SUBSPACEAD_SRC=/path/to/SubspaceAD   # SubspaceAD has no public repository
./scripts/install_methods.sh                # clones the other seven into third_party/
```

Optional extras used by the paper's experiments:

```bash
python -m pip install -e ".[backbone]"   # timm (DINOv3 embedding extraction)
python -m pip install -e ".[excel]"      # openpyxl (xlsx rendering, optional)
```

## Data layout

`GOEF_DATA` must contain the two standard benchmarks in their usual form:

```
$GOEF_DATA/mvtec_anomaly_detection/<category>/{train/good,test/<defect>,ground_truth/<defect>}
$GOEF_DATA/VisA_pytorch/1cls/<category>/{train/good,test,ground_truth}
```

Download: [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) ·
[VisA](https://github.com/amazon-science/spot-diff).  The VisA `1cls` layout is
the one produced by `tools/prepare_visa.py` of the original VisA release.

## Reproduction

```bash
export GOEF_WORKSPACE=/data/goef-work      # scratch space for selections/centres/results
export GOEF_DATA=/data/datasets
export GOEF_DINOV3=/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

./scripts/run_all.sh
```

`run_all.sh` runs seven idempotent stages; each can be run on its own:

| Stage | Command | Produces |
|---|---|---|
| 1 | `01_select_representatives.sh` | `sampling/<method>/<Dataset>_k5_seed<seed>/summary.csv` |
| 2 | `02_build_centers.sh` | `centers/<shot>-shot/<center>/<dataset>/` (62 datasets) |
| 3 | `03_compute_weights.sh` | `sampling/weights/{mvtec,visa}_k<shot>.csv` |
| 4 | `04_run_one_shot_sweeps.sh` | one row per reference image, 5 methods |
| 5 | `05_run_center_shots.sh` | one CSV per method, shot and centre, 8 methods |
| 6 | `06_aggregate_validation.sh` | Table I |
| 7 | `07_aggregate_reevaluation.sh` | Table II |

Stages 4 and 5 dominate the cost: stage 4 evaluates 12 288 reference images per
method (5 × 27 categories), stage 5 evaluates 62 centre datasets per method and
trains INP-Former, PromptAD, FoundAD and PatchCore from scratch.

A single setting can be reproduced on its own, e.g. 1-shot Table II:

```bash
./scripts/05_run_center_shots.sh 1
ONLY=patchcore ./scripts/05_run_center_shots.sh 1
```

## Results

### Table I — empirical validation of the evaluation strategies

`ϵ̄` is the average approximation error `|μ_g − μ_r|` (×100) and `r̄` the average
rank of `ε` among the six strategies, over 2 datasets × 5 detectors × 5 seeds ×
27 categories × 4 metrics = 2 700 comparisons per strategy.

| Metric | SINGLE | RE-RS | RE-CS | GOEF-KC | GOEF-KD | GOEF-KM |
|---|---|---|---|---|---|---|
| `ϵ̄ (×100)` | 1.617 | 0.681 | 0.778 | 0.752 | 0.543 | **0.486** |
| `r̄` | 4.47 | 3.16 | 3.50 | 3.48 | 3.00 | **2.94** |

Reproduced by `scripts/06_aggregate_validation.sh`; the exact table is
`results/validation/table_validation.csv`.  See
[`docs/results.md`](docs/results.md) for the discussion of the 0.01 difference
in the `SINGLE` rank.

### Table II — re-evaluation of eight FS-IAD methods

1-shot, `Image AUROC`:

| Method | MVTec-AD | VisA |
|---|---|---|
| PatchCore | 83.64 | 77.07 |
| WinCLIP | 93.42 | 81.63 |
| PromptAD | 93.57 | 86.82 |
| AnomalyDINO | **96.42** | 89.60 |
| FoundAD | 94.68 | 92.01 |
| INP-Former | 95.43 | 93.08 |
| UniVAD | 96.29 | **93.62** |
| SubspaceAD | 95.68 | 91.99 |

The complete 1/2/4-shot × four-metric table is
`results/reevaluation/all_final_weighted_results.csv`, and
[`docs/results.md`](docs/results.md) renders it in the paper's layout.

## Using GOEF for your own method

Evaluation only needs the method to expose a k-shot entry point, i.e. to accept
a dataset directory whose `train/good` contains the k reference images:

```bash
# 1. build the k-shot centre datasets once
./scripts/02_build_centers.sh

# 2. run your method on every centre and write one CSV per centre with
#    Category, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO
#    (see evaluation/README.md for the 11-column unified schema)

# 3. aggregate with the GOEF weights
python -m goef.cli.reevaluate \
  --method MyMethod=/path/to/MyMethod \
  --sampling_root "$GOEF_WORKSPACE/sampling" \
  --output results/my_method
```

The metric protocol is `evaluation/common/unified_evaluation.py`; copy it into
your project so that the four numbers remain comparable with Table II.

## Documentation

* [`docs/methodology.md`](docs/methodology.md) — every equation of the paper and the function that implements it
* [`docs/pipeline.md`](docs/pipeline.md) — stage-by-stage data flow and file schemas
* [`docs/datasets.md`](docs/datasets.md) — benchmark layout, centre datasets, weight semantics
* [`docs/results.md`](docs/results.md) — Table I / Table II reproduced, plus the known numerical drift
* [`evaluation/README.md`](evaluation/README.md) — the thirteen method adapters
* [`docs/third_party.md`](docs/third_party.md) — upstream revisions and licences

## Citation

```bibtex
@article{wang2026goef,
  title   = {Don't Miss Forest for A Tree: The Globality-Orientated Evaluation
             Framework for Few-shot Image Anomaly Detection},
  author  = {Wang, Siqi and Wang, Shiru and Xian, Rundong and Wang, Guangpu and
             He, Sha and Xu, Chuanfu and Liu, Jie and Liu, Xinwang},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026}
}
```

## Licence

MIT for the code in `goef/`, `scripts/`, `configs/`, `tests/` and the adapters
written for this paper. The files under `evaluation/**/files/` that are derived
from upstream projects keep their original licences; see
[`docs/third_party.md`](docs/third_party.md).
