# Third-party projects

The paper evaluates eight FS-IAD methods.  Their upstream repositories are not
vendored here: this repository ships **only the files that differ** from those
projects (`../modifications/`) and a script that clones the upstream sources and
applies the overlay (`../scripts/install_methods.sh`).

| Method | Upstream | Licence | Seed / notes |
|---|---|---|---|
| PatchCore | <https://github.com/amazon-science/patchcore-inspection> | Apache-2.0 | WideResNet-50, layers 2+3, coreset 10 % |
| WinCLIP | <https://github.com/zqhang/Accurate-WinCLIP-pytorch> | MIT | `ViT-B-16-plus-240`, 240 px |
| AnomalyDINO | <https://github.com/dammsi/AnomalyDINO> | Apache-2.0 | DINOv2 ViT-S/14, 672 px, training-free |
| INP-Former | <https://github.com/luow23/INPFormer> | MIT | DINOv2 ViT-B/14, 448→392, 200 epochs |
| SubspaceAD | <https://github.com/CLendering/SubspaceAD> | no licence file upstream | DINOv2-g, 518 px, `--layers -12…-18` |
| PromptAD | <https://github.com/FuNz-0/PromptAD> | Unlicense (public domain) | CLIP ViT-B/16, 100 epochs |
| FoundAD | <https://github.com/ymxlzgy/FoundAD> | no licence file upstream | DINOv3 projector, pre-trained checkpoints |
| UniVAD | <https://github.com/FantasticGNU/UniVAD> | CC BY-NC-SA 4.0 (non-commercial) | SAM-HQ + RAM + GroundingDINO, training-free |

Where the upstream project publishes no licence (SubspaceAD, FoundAD), the
modified files are provided **for research reproducibility only** and are not
covered by this repository's MIT licence.  UniVAD is non-commercial.

## Installation

```bash
# both experiments (separate checkouts, see below)
./scripts/install_methods.sh

# only one experiment
./scripts/install_methods.sh validation        # Sec. III-D
./scripts/install_methods.sh reevaluation      # Sec. IV

# only one project
METHOD=SubspaceAD ./scripts/install_methods.sh reevaluation
```

The result is

```
third_party/
├── validation/<project>/       # per-reference sweeps
└── reevaluation/<project>/     # per-centre evaluations
```

**Two checkouts are required** for PatchCore, WinCLIP, AnomalyDINO and
SubspaceAD: the two experiments modify the same upstream files in incompatible
ways (`--train_image`, `specific_image_paths`, `ref_sample_names`, `--k_shot_ref`).

To apply the files by hand:

```bash
cp -r modifications/<method>/<experiment>/files/. /path/to/<upstream>/
cp    modifications/common/unified_evaluation.py /path/to/<upstream>/
```

## What is modified

Full per-file lists and run commands for each project are in
[`../modifications/README.md`](../modifications/README.md) and
`modifications/<method>/README.md`.  In summary:

### New files (written for this paper, MIT)

* per-reference sweep drivers for PatchCore, WinCLIP, INP-Former, SubspaceAD and
  AnomalyDINO (`modifications/<method>/validation/files/`);
* per-centre evaluation drivers for all eight methods
  (`modifications/<method>/reevaluation/files/`);
* `unified_evaluation.py`, the shared metric and benchmark module, copied to the
  root of every project (`modifications/common/`).

### Modified upstream files (kept verbatim, under the upstream licence)

| Project | Files | Why |
|---|---|---|
| PatchCore | `bin/run_patchcore.py`, `bin/load_and_evaluate_patchcore.py`, `src/patchcore/datasets/mvtec.py`, `src/patchcore/metrics.py`, `src/patchcore/patchcore.py`, `src/patchcore/common.py` | `--train_image` filter, AUPRO helper, benchmark hooks, unified CSV |
| WinCLIP | `reproduce_WinCLIP.py`, `dataset.py`, `few_shot.py`, `data/{mvtec,visa}.py`, `src/open_clip/modified_resnet.py`, `src/training/params.py` | single-support evaluation with cached test features, preset training split, seed 42 |
| AnomalyDINO | `run_anomalydino.py`, `run_anomalydino_batched.py`, `src/{detection,backbones,post_eval}.py` | named references, reference-count validation, unified metrics |
| INP-Former | `INP_Former_{Few_Shot,Single_Class,Multi_Class,Super_Multi_Class,Zero_Shot}.py`, `utils.py`, seed defaults | `--shot`, `--result_csv`, unified metrics |
| SubspaceAD | `main.py`, `src/subspacead/{config.py,core/extractor.py,core/patching.py,utils/viz.py}` | `--k_shot_ref`, DINOv3 backbones, missing-mask tolerance, unified CSV |
| PromptAD | `train_cls.py`, `train_seg.py`, `test_cls.py`, `test_seg.py`, `datasets/*.py`, `utils/*.py` | centre `meta.json` layout, per-class CSV, GPU-peak logging |
| FoundAD | `foundad/main.py`, `foundad/src/{AD,foundad,train}.py`, `foundad/src/datasets/dataset.py`, configs, seed defaults | deterministic seed, unified metrics, centre layout |
| UniVAD | `UniVAD.py`, `test_univad.py`, `test_univad_no_pixel.py` | external reference images, mask paths, `--output_csv` |

## Environments

The eight methods need incompatible dependency sets.  The paper used one conda
environment per method:

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

## Pinning revisions

`install_methods.sh` clones the default branch.  For an exact reproduction, pin
the revisions you used:

```bash
PIN_PATCHCORE=<sha> PIN_WINCLIP=<sha> PIN_ANOMALYDINO=<sha> PIN_INPFORMER=<sha> \
PIN_SUBSPACEAD=<sha> PIN_PROMPTAD=<sha> PIN_FOUNDAD=<sha> PIN_UNIVAD=<sha> \
./scripts/install_methods.sh
```

The modifications were produced against the upstream snapshots of
October 2026.

## Metric comparability

Each upstream project has its own AUPRO implementation and its own pixel
resolution (224 PatchCore, 240 WinCLIP, 256 INP-Former, 512/518 SubspaceAD, 672
AnomalyDINO, …).  The overlay replaces every project's metric code with
`unified_evaluation.py`, so the four columns of Table II follow a single
protocol; the method-specific resolutions remain those recommended by each
paper, as stated in Sec. IV-A.
