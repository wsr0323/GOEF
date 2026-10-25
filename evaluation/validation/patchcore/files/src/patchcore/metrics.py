"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from skimage import measure


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    average_precision = metrics.average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }


def compute_aupro(anomaly_segmentations, ground_truth_masks, max_fpr=0.3):
    """Compute the standard per-region-overlap area up to ``max_fpr``."""
    segmentations = np.asarray(anomaly_segmentations)
    masks = np.asarray(ground_truth_masks).astype(bool)
    if segmentations.ndim == 4 and segmentations.shape[1] == 1:
        segmentations = segmentations[:, 0]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    normal_pixels = ~masks
    if not masks.any() or not normal_pixels.any():
        return float("nan")

    regions = [measure.regionprops(measure.label(mask)) for mask in masks]
    score_min, score_max = float(segmentations.min()), float(segmentations.max())
    if score_min == score_max:
        return 0.0

    fprs, pros = [0.0], [0.0]
    for threshold in np.linspace(score_max, score_min, num=200):
        prediction = segmentations >= threshold
        fpr = np.logical_and(prediction, normal_pixels).sum() / normal_pixels.sum()
        if fpr > max_fpr:
            continue
        overlaps = []
        for prediction_i, regions_i in zip(prediction, regions):
            for region in regions_i:
                coords = region.coords
                overlaps.append(prediction_i[coords[:, 0], coords[:, 1]].mean())
        if overlaps:
            fprs.append(fpr)
            pros.append(float(np.mean(overlaps)))

    fprs, pros = np.asarray(fprs), np.asarray(pros)
    order = np.argsort(fprs)
    unique_fprs, inverse = np.unique(fprs[order], return_inverse=True)
    max_pros = np.zeros_like(unique_fprs)
    np.maximum.at(max_pros, inverse, pros[order])
    if len(unique_fprs) < 2:
        return 0.0
    return float(metrics.auc(unique_fprs / max_fpr, max_pros))


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
