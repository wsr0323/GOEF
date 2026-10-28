"""Shared anomaly-detection metrics and inference benchmark utilities.

Copy this file unchanged into every benchmarked project.  Keeping metric and
timing logic here prevents project-specific evaluation code from silently
changing the comparison protocol.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label
from sklearn.metrics import average_precision_score, roc_auc_score


AUPRO_FPR_LIMIT = 0.30
AUPRO_NUM_THRESHOLDS = 200

RESULT_COLUMNS = (
    "Category",
    "Image AUROC",
    "Image AUPR",
    "Pixel AUROC",
    "Pixel AUPRO",
    "Images",
    "Inference Total (s)",
    "Inference (ms/image)",
    "Throughput (images/s)",
    "GPU Peak Allocated (MiB)",
    "GPU Peak Reserved (MiB)",
)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_maps(values: Any, name: str) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        arrays = [_as_numpy(value).squeeze() for value in values]
        if not arrays:
            raise ValueError(f"{name} must not be empty")
        result = np.stack(arrays, axis=0)
    else:
        result = np.squeeze(_as_numpy(values))
        if result.ndim == 2:
            result = result[None]
    if result.ndim != 3:
        raise ValueError(f"{name} must have shape [N, H, W], got {result.shape}")
    return result


def _safe_score(metric_name: str, labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError(
            f"{metric_name}: label/score shape mismatch: {labels.shape} vs {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{metric_name}: scores contain NaN or infinity")
    if np.unique(labels).size < 2:
        return float("nan")
    if metric_name.endswith("AUROC"):
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def compute_aupro(
    masks: Any,
    anomaly_maps: Any,
    fpr_limit: float = AUPRO_FPR_LIMIT,
    num_thresholds: int = AUPRO_NUM_THRESHOLDS,
) -> float:
    """Compute per-region-overlap AUC over global background FPR [0, limit].

    Masks use 8-connected components. Thresholds are sampled uniformly in FPR
    space from background-score quantiles, making the approximation invariant
    to monotonic score scaling. The area is divided by ``fpr_limit``.
    """

    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    mask_array = _stack_maps(masks, "masks") > 0
    score_array = _stack_maps(anomaly_maps, "anomaly_maps").astype(np.float64)
    if mask_array.shape != score_array.shape:
        raise ValueError(
            f"mask/anomaly-map shape mismatch: {mask_array.shape} vs {score_array.shape}"
        )
    if not np.isfinite(score_array).all():
        raise ValueError("anomaly_maps contain NaN or infinity")

    background_scores = np.sort(score_array[~mask_array])
    if background_scores.size == 0:
        return float("nan")

    structure = generate_binary_structure(2, 2)
    region_scores = []
    for mask, scores in zip(mask_array, score_array):
        components, component_count = label(mask, structure=structure)
        for component_id in range(1, component_count + 1):
            region_scores.append(np.sort(scores[components == component_id]))
    if not region_scores:
        return float("nan")

    target_fprs = np.linspace(0.0, fpr_limit, num_thresholds + 1)
    quantile_indices = np.floor(
        (1.0 - target_fprs) * (background_scores.size - 1)
    ).astype(np.int64)
    quantile_thresholds = background_scores[quantile_indices]
    thresholds = np.unique(
        np.concatenate(
            (
                [np.inf],
                quantile_thresholds,
                np.nextafter(quantile_thresholds, -np.inf),
            )
        )
    )

    false_positive_rates = np.empty(thresholds.size, dtype=np.float64)
    per_region_overlaps = np.zeros(thresholds.size, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        false_positives = background_scores.size - np.searchsorted(
            background_scores, threshold, side="right"
        )
        false_positive_rates[index] = false_positives / background_scores.size
        for scores in region_scores:
            detected = scores.size - np.searchsorted(scores, threshold, side="right")
            per_region_overlaps[index] += detected / scores.size
    per_region_overlaps /= len(region_scores)

    order = np.argsort(false_positive_rates, kind="stable")
    false_positive_rates = false_positive_rates[order]
    per_region_overlaps = per_region_overlaps[order]

    # A lower threshold can improve PRO without changing FPR. Retain the best
    # overlap for each repeated FPR before trapezoidal integration.
    unique_fprs, starts = np.unique(false_positive_rates, return_index=True)
    unique_pros = np.maximum.reduceat(per_region_overlaps, starts)
    unique_pros = np.maximum.accumulate(unique_pros)

    below = unique_fprs < fpr_limit
    integration_fprs = unique_fprs[below]
    integration_pros = unique_pros[below]
    boundary_pro = np.interp(fpr_limit, unique_fprs, unique_pros)
    integration_fprs = np.append(integration_fprs, fpr_limit)
    integration_pros = np.append(integration_pros, boundary_pro)
    if integration_fprs[0] > 0.0:
        integration_fprs = np.insert(integration_fprs, 0, 0.0)
        integration_pros = np.insert(integration_pros, 0, 0.0)

    return float(np.trapz(integration_pros, integration_fprs) / fpr_limit)


def compute_metrics(
    image_labels: Any,
    image_scores: Any,
    pixel_masks: Any,
    anomaly_maps: Any,
    fpr_limit: float = AUPRO_FPR_LIMIT,
) -> Dict[str, float]:
    """Compute the four standardized metrics for one dataset category."""

    image_metrics = compute_image_metrics(image_labels, image_scores)
    pixel_metrics = compute_pixel_metrics(pixel_masks, anomaly_maps, fpr_limit)
    labels = _as_numpy(image_labels).reshape(-1)
    masks = _stack_maps(pixel_masks, "pixel_masks")
    if labels.size != masks.shape[0]:
        raise ValueError(
            f"image/pixel sample count mismatch: {labels.size} vs {masks.shape[0]}"
        )
    return {**image_metrics, **pixel_metrics}


def compute_image_metrics(image_labels: Any, image_scores: Any) -> Dict[str, float]:
    """Compute image AUROC and average precision (reported as Image AUPR)."""

    labels = (_as_numpy(image_labels).reshape(-1) > 0).astype(np.uint8)
    scores = _as_numpy(image_scores).reshape(-1).astype(np.float64)
    return {
        "Image AUROC": _safe_score("Image AUROC", labels, scores),
        "Image AUPR": _safe_score("Image AUPR", labels, scores),
    }


def compute_pixel_metrics(
    pixel_masks: Any,
    anomaly_maps: Any,
    fpr_limit: float = AUPRO_FPR_LIMIT,
) -> Dict[str, float]:
    """Compute global pixel AUROC and 8-connected Pixel AUPRO."""

    masks = _stack_maps(pixel_masks, "pixel_masks") > 0
    maps = _stack_maps(anomaly_maps, "anomaly_maps").astype(np.float64)
    if masks.shape != maps.shape:
        raise ValueError(f"mask/anomaly-map shape mismatch: {masks.shape} vs {maps.shape}")

    return {
        "Pixel AUROC": _safe_score("Pixel AUROC", masks.reshape(-1), maps.reshape(-1)),
        "Pixel AUPRO": compute_aupro(masks, maps, fpr_limit=fpr_limit),
    }


class InferenceBenchmark:
    """CUDA-synchronized batch timer and PyTorch GPU peak-memory recorder."""

    def __init__(self, device: Any) -> None:
        self.device = torch.device(device)
        self.cuda_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.total_seconds = 0.0
        self.image_count = 0
        self._started_at = None
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_batch(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("benchmark batch timer is already running")
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
        self._started_at = time.perf_counter()

    def stop_batch(self, image_count: int) -> float:
        if self._started_at is None:
            raise RuntimeError("benchmark batch timer was not started")
        if image_count < 0:
            raise ValueError("image_count must be non-negative")
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._started_at
        self._started_at = None
        self.total_seconds += elapsed
        self.image_count += int(image_count)
        return elapsed

    def result(self) -> Dict[str, float]:
        if self._started_at is not None:
            raise RuntimeError("cannot report benchmark while a batch timer is running")
        milliseconds = (
            1000.0 * self.total_seconds / self.image_count
            if self.image_count
            else float("nan")
        )
        throughput = (
            self.image_count / self.total_seconds
            if self.total_seconds > 0.0
            else float("nan")
        )
        if self.cuda_enabled:
            allocated = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            reserved = torch.cuda.max_memory_reserved(self.device) / (1024.0**2)
        else:
            allocated = reserved = 0.0
        return {
            "Images": self.image_count,
            "Inference Total (s)": self.total_seconds,
            "Inference (ms/image)": milliseconds,
            "Throughput (images/s)": throughput,
            "GPU Peak Allocated (MiB)": allocated,
            "GPU Peak Reserved (MiB)": reserved,
        }


def make_result_row(
    category: str,
    metrics: Mapping[str, float],
    benchmark: Mapping[str, float],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"Category": category}
    row.update(metrics)
    row.update(benchmark)
    missing = [column for column in RESULT_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"result row is missing columns: {missing}")
    return {column: row[column] for column in RESULT_COLUMNS}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty result list")
    total_images = int(sum(int(row["Images"]) for row in rows))
    total_seconds = float(sum(float(row["Inference Total (s)"]) for row in rows))

    summary: Dict[str, Any] = {"Category": "Overall"}
    for metric in ("Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
    summary.update(
        {
            "Images": total_images,
            "Inference Total (s)": total_seconds,
            "Inference (ms/image)": (
                1000.0 * total_seconds / total_images if total_images else float("nan")
            ),
            "Throughput (images/s)": (
                total_images / total_seconds if total_seconds > 0.0 else float("nan")
            ),
            "GPU Peak Allocated (MiB)": max(
                float(row["GPU Peak Allocated (MiB)"]) for row in rows
            ),
            "GPU Peak Reserved (MiB)": max(
                float(row["GPU Peak Reserved (MiB)"]) for row in rows
            ),
        }
    )
    return {column: summary[column] for column in RESULT_COLUMNS}


def write_results_csv(
    output_path: Any,
    rows: Iterable[Mapping[str, Any]],
    include_summary: bool = True,
) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty result list")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = rows + ([summarize_rows(rows)] if include_summary else [])
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_path


def format_result(row: Mapping[str, Any]) -> str:
    return (
        f"{row['Category']} | Image AUROC {row['Image AUROC']:.4f} | "
        f"Image AUPR {row['Image AUPR']:.4f} | Pixel AUROC {row['Pixel AUROC']:.4f} | "
        f"Pixel AUPRO {row['Pixel AUPRO']:.4f} | "
        f"{row['Inference (ms/image)']:.3f} ms/image | "
        f"GPU peak {row['GPU Peak Allocated (MiB)']:.1f} MiB allocated"
    )
