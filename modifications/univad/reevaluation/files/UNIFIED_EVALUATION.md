# Unified anomaly-detection evaluation protocol

All eight projects use the same `unified_evaluation.py` implementation and the
same `unified_results*.csv` schema.

## Metrics

- **Image AUROC**: `sklearn.metrics.roc_auc_score` over all images in one
  category. Label `1` means anomalous.
- **Image AUPR**: `sklearn.metrics.average_precision_score` over the same image
  labels and scores. This is average precision, not trapezoidal PR-AUC.
- **Pixel AUROC**: `roc_auc_score` after globally flattening every binary mask
  and anomaly map in one category.
- **Pixel AUPRO**: mean per-region overlap using 8-connected ground-truth
  regions and global background-pixel FPR. Thresholds are sampled uniformly in
  FPR space, the curve is integrated over FPR `[0, 0.30]`, and the area is
  divided by `0.30`.

The `Overall` row is a macro mean of category metrics. Undefined metrics are
written as `nan` and ignored by the macro mean.

## Inference timing

The timer starts after an image or batch has been read by the dataset loader.
It includes input preparation/device transfer, model forward, anomaly-map
generation, image-score aggregation, and copying predictions to CPU. It
excludes training/reference-bank construction, dataset file I/O, metric
calculation, visualization, and output-file writes.

CUDA is synchronized immediately before and after each timed batch. No warm-up
batch is run by default, so the first batch is included consistently. Results
contain total seconds, milliseconds per image, and images per second for every
category. The `Overall` timing is recomputed from total images and total time.

## GPU memory

CUDA peak statistics are reset immediately before each category inference
loop. `GPU Peak Allocated (MiB)` and `GPU Peak Reserved (MiB)` come from the
PyTorch CUDA allocator; the `Overall` row reports the maximum category value.
Run test-only modes in an otherwise idle process for comparable results.
