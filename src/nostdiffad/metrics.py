from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.int64)
    if np.unique(labels).size < 2:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    idx = int(np.nanargmax(f1))
    if idx >= len(thresholds):
        threshold = float(thresholds[-1]) if len(thresholds) else float("nan")
    else:
        threshold = float(thresholds[idx])
    return float(f1[idx]), threshold


def iou_and_dice(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = scores > threshold
    target = labels > 0
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (pred_sum + target_sum) if (pred_sum + target_sum) else float("nan")
    return float(iou), float(dice)


def au_pro(mask_true: np.ndarray, score_maps: np.ndarray, steps: int = 100, max_fpr: float = 0.3) -> float:
    """Compute AU-PRO over connected foreground regions up to ``max_fpr``.

    The PRO curve is sampled on a uniform background-FPR grid. For every ground-truth
    connected component, the overlap is its TPR at the score threshold that yields the
    requested background FPR. This matches the common AUPRO protocol while avoiding
    score-spaced threshold bias.
    """

    try:
        from scipy import ndimage
    except Exception:
        return float("nan")

    masks = np.asarray(mask_true) > 0
    scores = np.asarray(score_maps, dtype=np.float64)
    if masks.shape != scores.shape:
        raise ValueError(f"mask_true and score_maps must have the same shape, got {masks.shape} and {scores.shape}.")

    background_scores = scores[~masks]
    if background_scores.size == 0:
        return float("nan")
    components: list[np.ndarray] = []
    for mask, score_map in zip(masks, scores):
        labeled, count = ndimage.label(mask)
        for comp_id in range(1, count + 1):
            comp_scores = score_map[labeled == comp_id]
            if comp_scores.size:
                components.append(np.sort(comp_scores.reshape(-1)))
    if not components:
        return float("nan")

    max_fpr = float(max_fpr)
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}.")
    fpr_grid = np.linspace(0.0, max_fpr, max(2, int(steps)))

    bg_sorted_desc = np.sort(background_scores.reshape(-1))[::-1]
    bg_count = bg_sorted_desc.size
    thresholds = np.empty_like(fpr_grid)
    thresholds[0] = np.nextafter(bg_sorted_desc[0], np.inf)
    if thresholds.size > 1:
        fp_counts = np.ceil(fpr_grid[1:] * bg_count).astype(np.int64)
        fp_counts = np.clip(fp_counts, 1, bg_count)
        thresholds[1:] = bg_sorted_desc[fp_counts - 1]

    pro_values = np.zeros_like(fpr_grid, dtype=np.float64)
    for comp_scores in components:
        true_positive_counts = comp_scores.size - np.searchsorted(comp_scores, thresholds, side="right")
        pro_values += true_positive_counts / max(comp_scores.size, 1)
    pro_values /= len(components)
    pro_values = np.maximum.accumulate(pro_values)
    return float(np.trapezoid(pro_values, fpr_grid) / max_fpr)


def calibration_error(normal_scores: np.ndarray, threshold: float, target_coverage: float) -> float:
    coverage = float((normal_scores <= threshold).mean()) if normal_scores.size else float("nan")
    return abs(coverage - target_coverage)


def compute_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    score_maps: np.ndarray,
    threshold: float | None,
    inference_seconds: float,
    normal_calibration_scores: np.ndarray | None = None,
    target_coverage: float = 0.95,
    au_pro_steps: int = 100,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "image_auroc": _safe_auc(image_labels, image_scores),
        "image_ap": _safe_ap(image_labels, image_scores),
        "pixel_auroc": _safe_auc(masks.reshape(-1), score_maps.reshape(-1)),
        "pixel_ap": _safe_ap(masks.reshape(-1), score_maps.reshape(-1)),
        "au_pro": au_pro(masks, score_maps, steps=au_pro_steps),
        "inference_speed_fps": float(len(image_labels) / max(inference_seconds, 1e-12)),
    }
    f1, best_threshold = f1_max(masks.reshape(-1), score_maps.reshape(-1))
    metrics["f1_max"] = f1
    threshold_for_overlap = best_threshold if threshold is None else threshold
    iou, dice = iou_and_dice(masks.reshape(-1), score_maps.reshape(-1), threshold_for_overlap)
    metrics["iou"] = iou
    metrics["dice"] = dice
    if threshold is not None and normal_calibration_scores is not None:
        metrics["calibration_error"] = calibration_error(normal_calibration_scores, threshold, target_coverage)
    else:
        metrics["calibration_error"] = float("nan")
    selection_terms = [
        (0.4, metrics["image_auroc"]),
        (0.4, metrics["pixel_auroc"]),
        (0.2, metrics["au_pro"]),
    ]
    valid_terms = [(weight, value) for weight, value in selection_terms if np.isfinite(value)]
    metrics["selection_score"] = (
        float(sum(weight * value for weight, value in valid_terms) / sum(weight for weight, _ in valid_terms))
        if valid_terms
        else float("nan")
    )
    pro_terms = [
        (0.2, metrics["image_auroc"]),
        (0.3, metrics["pixel_auroc"]),
        (0.5, metrics["au_pro"]),
    ]
    valid_pro_terms = [(weight, value) for weight, value in pro_terms if np.isfinite(value)]
    metrics["selection_score_pro"] = (
        float(sum(weight * value for weight, value in valid_pro_terms) / sum(weight for weight, _ in valid_pro_terms))
        if valid_pro_terms
        else float("nan")
    )
    return metrics
