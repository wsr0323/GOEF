# One-shot PatchCore runner

`bin/run_one_shot_patchcore.py` trains and evaluates PatchCore once for every
normal image in `category/train/good`. It supports both the MVTec and VisA
directory layouts already present on this machine.

The environment must contain this repository's dependencies, including
`timm`, PyTorch, torchvision, FAISS, scikit-image, and a GPU-enabled FAISS build
when using `--gpu`.

Run every category in both datasets:

```bash
python bin/run_one_shot_patchcore.py --gpu 0
```

Run only MVTec's bottle class:

```bash
python bin/run_one_shot_patchcore.py \
  --datasets mvtec --classes bottle --gpu 0 \
  --output-csv results/mvtec_bottle_one_shot.csv
```

Run only VisA's candle class:

```bash
python bin/run_one_shot_patchcore.py \
  --datasets visa --classes candle --gpu 0 \
  --output-csv results/visa_candle_one_shot.csv
```

The output CSV is appended immediately after a successful training image. Its
key is `object + reference_index`, so rerunning exactly the same command skips
completed images and retries only incomplete or failed images. The rows include
`Run`, `ref_sample`, `reference_index`, `Image AUROC`, `Image AUPR`, `Pixel
AUROC`, and `Pixel AUPRO`, following the layout of `subspacemvtec.csv`.

Per-image PatchCore artifacts are removed after their metrics are safely
recorded. Add `--keep-run-results` to retain them under
`one_shot_patchcore_results/raw/`.

The model uses `--gpu 0` by default, while FAISS stays on CPU because this
repository declares `faiss-cpu`. When `faiss-gpu` is installed, add
`--faiss-on-gpu` to move nearest-neighbour search to the GPU too.
