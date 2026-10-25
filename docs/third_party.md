# Third-party projects

The paper evaluates eight FS-IAD methods.  Their upstream repositories are
**not** vendored here: `scripts/install_methods.sh` clones them and copies the
GOEF adapters from `evaluation/` into the checkout.

| Method | Upstream | Licence | Adapter files |
|---|---|---|---|
| PatchCore | [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) | Apache-2.0 | `evaluation/*/patchcore/` — new sweep driver + modified `run_patchcore.py`, `metrics.py`, `datasets/mvtec.py` |
| WinCLIP | [zqhang/Accurate-WinCLIP-pytorch](https://github.com/zqhang/Accurate-WinCLIP-pytorch) | MIT | `evaluation/*/winclip/` — new 1-category driver + modified `reproduce_WinCLIP.py` |
| AnomalyDINO | [dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO) | Apache-2.0 | `evaluation/*/anomalydino/` — new per-reference sweeps, no upstream edit |
| INP-Former | [luow23/INP-Former](https://github.com/luow23/INP-Former) | MIT | `evaluation/validation/inpformer/` — new self-contained sweep, no upstream edit |
| SubspaceAD | official code of the CVPR 2026 paper; no public repository at the time of writing | not stated upstream | `evaluation/*/subspacead/` — new drivers + modified `main.py` |
| PromptAD | [FuNz-0/PromptAD](https://github.com/FuNz-0/PromptAD) | Unlicense (public domain) | `evaluation/reevaluation/promptad/` — new centre/shot drivers, no upstream edit |
| FoundAD | [ymxlzgy/FoundAD](https://github.com/ymxlzgy/FoundAD) | not stated upstream | `evaluation/reevaluation/foundad/` — new batch driver, no upstream edit |
| UniVAD | [FantasticGNU/UniVAD](https://github.com/FantasticGNU/UniVAD) | CC BY-NC-SA 4.0 | `evaluation/reevaluation/univad/` — new centre runner and mask builder, no upstream edit |

## What the adapters change

### New files (written for this paper, MIT)

* per-reference sweep drivers for PatchCore, WinCLIP, INP-Former, SubspaceAD and
  AnomalyDINO (`evaluation/validation/*/files/`);
* per-centre evaluation drivers for all eight methods
  (`evaluation/reevaluation/*/files/`);
* `unified_evaluation.py`, the shared metric/benchmark module copied into every
  project (`evaluation/common/`).

### Modified upstream files (kept verbatim, under the upstream licence)

| File | Modification |
|---|---|
| `bin/run_patchcore.py` | new `--train_image` option, forwarded to the dataset |
| `src/patchcore/datasets/mvtec.py` | keep only `--train_image` in `train/good` |
| `src/patchcore/metrics.py` | add `compute_aupro()` |
| `reproduce_WinCLIP.py` | add `SingleClassEvaluator`, `evaluate_support()`, test-feature caching, `specific_image_paths=` support in `memory()` |
| `main.py` (SubspaceAD) | add `--k_shot_ref`, reference index parsing, `ref_sample`/`Run`/`参考索引` columns, CPU AUPRO |

Those five files are included **only** to make the paper reproducible.  They
remain the property of their authors and are not covered by this repository's
MIT licence.  If you redistribute this repository, check each upstream licence
first; UniVAD is non-commercial, and SubspaceAD/FoundAD did not ship a licence
file at the time of writing.

## Pinning revisions

`install_methods.sh` clones the default branch.  For an exact reproduction, pin
the revisions you used:

```bash
PIN_PATCHCORE=<sha> PIN_WINCLIP=<sha> PIN_ANOMALYDINO=<sha> \
PIN_INPFORMER=<sha> PIN_PROMPTAD=<sha> PIN_FOUNDAD=<sha> PIN_UNIVAD=<sha> \
./scripts/install_methods.sh
```

The adapters were developed against the upstream snapshots of October 2026.

## Environments

The eight methods need incompatible dependency sets (e.g. INP-Former pins
`torch 2.0.0+cu118`, the others run on newer CUDA builds).  The paper used one
conda environment per method:

| Method | Environment | Notable pins |
|---|---|---|
| PatchCore | `patchcore` | `faiss-cpu`, `timm`, `scikit-image` |
| WinCLIP | `winclip` | `open-clip-torch`, `timm` |
| AnomalyDINO | `anomalydino` | `faiss-cpu`, `dinov2_vits14` |
| INP-Former | `inpformer` | `torch 2.0.0+cu118`, `adeval==1.1.0` |
| SubspaceAD | `subspacead` | `anomalib`, DINOv2-g |
| PromptAD | `prompt_ad` | `open_clip_torch` |
| FoundAD | `foundad` | `pytorch-lightning`, `hydra-core` |
| UniVAD | `univad` | SAM-HQ, RAM, GroundingDINO |

This repository's own package only needs `numpy`, `scikit-learn`, `pillow`,
`pyyaml` (plus `timm` for the DINOv3 embedding stage).

## Metric comparability

Each upstream project has its own AUPRO implementation and its own pixel
resolution (224 PatchCore, 240 WinCLIP, 256 INP-Former, 512 SubspaceAD, 672
AnomalyDINO, …).  The adapters replace every project's metric code with
`unified_evaluation.py`, so the four columns of Table II follow a single
protocol; the method-specific resolutions remain those recommended by each
paper, as stated in Sec. IV-A.
