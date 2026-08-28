"""Metrics for co-expression and spatial-field preservation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def _arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must match [spots, genes]")
    if prediction.shape[0] < 2 or prediction.shape[1] < 1:
        raise ValueError("at least two spots and one gene are required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("metric inputs must be finite")
    return prediction, target


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2 or np.std(right) < 1e-12:
        return float("nan")
    if np.std(left) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def undirected_knn_edges(coordinates: np.ndarray, k: int) -> np.ndarray:
    """Return unique undirected edges from a coordinate k-NN graph."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("coordinates must be [spots >= 2, 2]")
    count = min(max(int(k), 1) + 1, len(coordinates))
    _, neighbors = cKDTree(coordinates).query(coordinates, k=count)
    neighbors = np.asarray(neighbors)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    edges = {
        (min(i, int(j)), max(i, int(j)))
        for i, row in enumerate(neighbors)
        for j in row
        if i != int(j)
    }
    if not edges:
        raise ValueError("the spatial graph contains no edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def coexpression_agreement(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compare gene-gene correlation matrices across spots."""
    prediction, target = _arrays(prediction, target)
    eligible = target.std(axis=0) >= 1e-8
    if eligible.sum() < 2:
        return {"pcc": float("nan"), "mae": float("nan"), "n_pairs": 0}
    prediction, target = prediction[:, eligible], target[:, eligible]
    target_z = (target - target.mean(0)) / np.maximum(target.std(0), 1e-12)
    pred_std = prediction.std(0)
    pred_z = (prediction - prediction.mean(0)) / np.maximum(pred_std, 1e-12)
    pred_z[:, pred_std < 1e-8] = 0.0
    upper = np.triu_indices(target.shape[1], k=1)
    true_pairs = (target_z.T @ target_z / len(target))[upper]
    pred_pairs = (pred_z.T @ pred_z / len(prediction))[upper]
    return {
        "pcc": _correlation(pred_pairs, true_pairs),
        "mae": float(np.mean(np.abs(pred_pairs - true_pairs))),
        "n_pairs": int(true_pairs.size),
    }


def _moran(expression: np.ndarray, edges: np.ndarray, chunk: int = 64) -> np.ndarray:
    left, right = edges[:, 0], edges[:, 1]
    output = np.full(expression.shape[1], np.nan, dtype=np.float64)
    for start in range(0, expression.shape[1], chunk):
        end = min(start + chunk, expression.shape[1])
        centered = expression[:, start:end] - expression[:, start:end].mean(0)
        denominator = np.sum(centered * centered, axis=0)
        numerator = 2.0 * np.sum(centered[left] * centered[right], axis=0)
        valid = denominator > 1e-12
        output[start:end][valid] = (
            len(expression) / (2.0 * len(edges)) * numerator[valid] / denominator[valid]
        )
    return output


def moran_agreement(
    prediction: np.ndarray, target: np.ndarray, coordinates: np.ndarray, k: int
) -> dict[str, float]:
    """Compare per-gene Moran's I on one common graph."""
    prediction, target = _arrays(prediction, target)
    edges = undirected_knn_edges(coordinates, k)
    truth = _moran(target, edges)
    predicted = _moran(prediction, edges)
    eligible = np.isfinite(truth)
    predicted[eligible & ~np.isfinite(predicted)] = 0.0
    paired = eligible & np.isfinite(predicted)
    return {
        "pcc": _correlation(predicted[paired], truth[paired]),
        "mae": float(np.mean(np.abs(predicted[paired] - truth[paired])))
        if paired.any()
        else float("nan"),
        "target_mean": float(np.mean(truth[paired])) if paired.any() else float("nan"),
        "prediction_mean": float(np.mean(predicted[paired]))
        if paired.any()
        else float("nan"),
        "n_genes": int(paired.sum()),
        "n_edges": int(len(edges)),
    }


def gradient_agreement(
    prediction: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    gene_scale: np.ndarray,
    k: int,
    threshold: float = 0.25,
) -> dict[str, float]:
    """Compare signed neighbor gradients in training-standard-deviation units."""
    prediction, target = _arrays(prediction, target)
    scale = np.asarray(gene_scale, dtype=np.float64)
    if scale.shape != (prediction.shape[1],) or np.any(scale <= 0):
        raise ValueError("gene_scale must be positive and align with genes")
    edges = undirected_knn_edges(coordinates, k)
    left, right = edges[:, 0], edges[:, 1]
    sums = np.zeros(8, dtype=np.float64)
    sign_matches = sign_count = 0
    gene_scores: list[np.ndarray] = []
    for start in range(0, prediction.shape[1], 32):
        end = min(start + 32, prediction.shape[1])
        block_scale = scale[None, start:end]
        pred = (prediction[right, start:end] - prediction[left, start:end]) / block_scale
        truth = (target[right, start:end] - target[left, start:end]) / block_scale
        error = pred - truth
        sums += np.asarray(
            [
                pred.size,
                pred.sum(),
                truth.sum(),
                np.square(pred).sum(),
                np.square(truth).sum(),
                (pred * truth).sum(),
                np.square(error).sum(),
                np.abs(error).sum(),
            ]
        )
        nontrivial = np.abs(truth) >= threshold
        sign_matches += int((np.signbit(pred[nontrivial]) == np.signbit(truth[nontrivial])).sum())
        sign_count += int(nontrivial.sum())
        gene_scores.append(
            np.asarray([_correlation(pred[:, i], truth[:, i]) for i in range(pred.shape[1])])
        )
    n, sum_p, sum_t, sum_p2, sum_t2, cross, square_error, abs_error = sums
    covariance = cross - sum_p * sum_t / n
    pred_ss = sum_p2 - sum_p * sum_p / n
    target_ss = sum_t2 - sum_t * sum_t / n
    global_pcc = covariance / np.sqrt(pred_ss * target_ss) if pred_ss > 1e-12 and target_ss > 1e-12 else 0.0
    per_gene = np.concatenate(gene_scores)
    per_gene = per_gene[np.isfinite(per_gene)]
    return {
        "pcc": float(np.clip(global_pcc, -1.0, 1.0)),
        "mean_gene_pcc": float(per_gene.mean()) if len(per_gene) else float("nan"),
        "rmse": float(np.sqrt(square_error / n)),
        "mae": float(abs_error / n),
        "energy_ratio": float(sum_p2 / sum_t2) if sum_t2 > 1e-12 else float("nan"),
        "sign_agreement": float(sign_matches / sign_count) if sign_count else float("nan"),
        "n_edges": int(len(edges)),
    }


def spatial_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    max_side: int = 512,
) -> dict[str, float]:
    """Calculate masked per-gene SSIM on a common spot-derived raster."""
    prediction, target = _arrays(prediction, target)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    nearest = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    pixel = max(float(np.median(nearest) / 2.0), 1e-12)
    span = np.ptp(coordinates, axis=0)
    pixel = max(pixel, float(np.max(span) / max(max_side - 11, 1)))
    xy = np.rint((coordinates - coordinates.min(0)) / pixel).astype(np.int64) + 5
    columns, rows = xy[:, 0], xy[:, 1]
    shape = (int(rows.max()) + 6, int(columns.max()) + 6)
    impulses = np.zeros(shape, dtype=np.float64)
    np.add.at(impulses, (rows, columns), 1.0)
    weight = gaussian_filter(impulses, 1.0, mode="constant", truncate=3.0)
    support = weight > max(weight.max() * 1e-3, 1e-12)
    local_weight = np.maximum(gaussian_filter(support.astype(float), 1.5), 1e-12)
    scores = np.full(prediction.shape[1], np.nan)
    chunk = max(1, min(32, 2_000_000 // (shape[0] * shape[1])))
    for start in range(0, prediction.shape[1], chunk):
        end = min(start + chunk, prediction.shape[1])
        width = end - start
        pred_impulse = np.zeros((*shape, width))
        true_impulse = np.zeros((*shape, width))
        np.add.at(pred_impulse, (rows, columns), prediction[:, start:end])
        np.add.at(true_impulse, (rows, columns), target[:, start:end])
        denominator = np.maximum(weight[:, :, None], 1e-12)
        mask = support[:, :, None]
        pred = np.where(mask, gaussian_filter(pred_impulse, (1, 1, 0)) / denominator, 0)
        truth = np.where(mask, gaussian_filter(true_impulse, (1, 1, 0)) / denominator, 0)
        local_denominator = local_weight[:, :, None]
        pred_mean = gaussian_filter(pred * mask, (1.5, 1.5, 0)) / local_denominator
        true_mean = gaussian_filter(truth * mask, (1.5, 1.5, 0)) / local_denominator
        pred_var = np.maximum(
            gaussian_filter(pred * pred * mask, (1.5, 1.5, 0)) / local_denominator - pred_mean**2,
            0,
        )
        true_var = np.maximum(
            gaussian_filter(truth * truth * mask, (1.5, 1.5, 0)) / local_denominator - true_mean**2,
            0,
        )
        covariance = (
            gaussian_filter(pred * truth * mask, (1.5, 1.5, 0)) / local_denominator
            - pred_mean * true_mean
        )
        data_range = np.ptp(target[:, start:end], axis=0)
        c1 = (0.01 * np.maximum(data_range, 1e-8))[None, None, :] ** 2
        c2 = (0.03 * np.maximum(data_range, 1e-8))[None, None, :] ** 2
        score_map = (
            (2 * pred_mean * true_mean + c1) * (2 * covariance + c2)
            / np.maximum((pred_mean**2 + true_mean**2 + c1) * (pred_var + true_var + c2), 1e-18)
        )
        block = np.mean(score_map[rows, columns], axis=0)
        block[data_range < 1e-8] = np.nan
        scores[start:end] = np.clip(block, -1, 1)
    finite = scores[np.isfinite(scores)]
    return {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "n_genes": int(len(finite)),
    }


def structured_panel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    gene_scale: np.ndarray,
    panels: Mapping[str, Sequence[int]],
    local_k: int = 6,
    wide_k: int = 18,
) -> dict[str, dict]:
    """Calculate structured metrics for fixed training-derived panels."""
    result: dict[str, dict] = {}
    for name, raw_indices in panels.items():
        indices = np.asarray(tuple(raw_indices), dtype=np.int64)
        pred, truth, scale = prediction[:, indices], target[:, indices], gene_scale[indices]
        entry = {
            "moran": moran_agreement(pred, truth, coordinates, local_k),
            "gradient_local": gradient_agreement(pred, truth, coordinates, scale, local_k),
            "gradient_wide": gradient_agreement(pred, truth, coordinates, scale, wide_k),
        }
        # Full-transcriptome co-expression and SSIM are needlessly quadratic/heavy.
        if len(indices) <= 256:
            entry["coexpression"] = coexpression_agreement(pred, truth)
            entry["spatial_ssim"] = spatial_ssim(pred, truth, coordinates)
        result[name] = entry
    return result
