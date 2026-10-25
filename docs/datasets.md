# Datasets, centre datasets and weights

## Benchmark layout

GOEF uses the two standard FS-IAD benchmarks in their 1-class layout.

### MVTec-AD

```
mvtec_anomaly_detection/
├── meta.json                      optional (built automatically if absent)
└── <category>/
    ├── train/good/*.png
    ├── test/<defect>/*.png
    └── ground_truth/<defect>/*_mask.png
```

15 categories, 3 629 normal training images.

### VisA (1cls)

```
VisA_pytorch/1cls/
└── <category>/
    ├── train/good/*.JPG
    ├── test/good/*.JPG
    ├── test/bad/*.JPG
    └── ground_truth/bad/*.png
```

12 categories, 8 659 normal training images.  VisA's original split lives in
`split_csv/1cls.csv`; the `1cls` directory is the "one class per directory"
export that the evaluated projects expect.

## What a "centre dataset" is

GOEF partitions each category's training set into `n = 5` clusters and picks one
representative image per cluster.  A **centre dataset** keeps exactly the
representatives of one cluster combination, for every category, and copies the
complete test split.  With `n = 5` there are

| shot k | directory | count |
|---|---|---|
| 1 | `center_01` … `center_05` | 5 |
| 2 | `centers_01_02` … `centers_04_05` | 10 |
| 3 | `centers_01_02_03` … `centers_03_04_05` | 10 |
| 4 | `centers_01_02_03_04` … `centers_02_03_04_05` | 5 |
| 5 | `centers_01_02_03_04_05` | 1 |

31 datasets per benchmark, 62 in total — the "62 evaluation tasks" of Sec. IV.

```
center_01/
├── meta.json            {"train": {"bottle": [1 entry], …}, "test": {…all test images…}}
├── representatives.csv  category,center_number,cluster_id,img_path,representative_artifact
├── dataset_info.json    provenance: source_root, cluster_root, materialization, counts
└── bottle/{train/good/164.png, test/…, ground_truth/…}
```

Every `meta.json` entry has the five fields the upstream projects expect:
`img_path`, `mask_path`, `cls_name`, `specie_name`, `anomaly`.

## Weight semantics

For a category with `N` training images split into clusters of size
`N_1 … N_n`, the k-shot weight of a centre set `τ` is

```
w_τ = N_τ / C(N, k),     N_τ = Σ_{Σ a_s = k} Π_s C(N_{τ_s}, a_s)
```

| Quantity | Meaning |
|---|---|
| `evaluation_shot = q = |τ|` | which centre dataset to run (q-shot) |
| `count = N_τ` | number of k-subsets represented by that dataset |
| `total_count = C(N,k)` | number of k-subsets in total |
| `weight` | `count / total_count` |

Properties enforced by `goef.weights` and by `tests/test_weights.py`:

* weights of a category sum to 1;
* `w_τ = 0` combinations (a cluster smaller than the required multiplicity) are
  omitted rather than written as zero;
* for `k = 1` the weights are exactly `N_j / N`, i.e. Eq. (2);
* the implementation reproduces the per-shot weight tables of the paper's
  pipeline for `k = 1…5` to `5e-13`.

### Example — MVTec-AD `bottle` (`N = 209`, clusters `54/37/57/24/37`)

| shot | centres | q | count | weight |
|---|---|---|---|---|
| 1 | `01` | 1 | 54 | 0.258373 |
| 1 | `04` | 1 | 24 | 0.114833 |
| 2 | `01_01` | 1 | C(54,2) = 1 431 | 0.065835 |
| 2 | `01_02` | 2 | 54·37 = 1 998 | 0.091921 |
| 4 | `01_02` | 2 | 2 290 374 | 0.029653 |
| 4 | `01_02_03_04` | 4 | 2 733 264 | 0.035387 |

`01_01` is a *repeated* centre: 2-shot evaluation with the same representative
twice is numerically the 1-shot experiment of `center_01`, so its weight is
folded into the 1-shot result.  This is exactly the Algorithm 2 line 15 reuse.

## Cost

| Item | Size |
|---|---|
| 62 centre datasets (`copy`) | ≈ 2× the two benchmarks |
| per-reference sweeps (5 methods) | 12 288 evaluations each |
| per-centre runs (8 methods) | 62 evaluations each per shot |
| selection artefacts (`summary.csv` only) | < 1 MB per run |

Use `--materialize symlink` (default) to avoid duplicating the images; only
methods that refuse symlinks need `copy`.
