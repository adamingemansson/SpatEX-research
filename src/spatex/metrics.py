"""Metrics for spatial expression prediction.

Gene-wise spatial PCC is the primary score. Flattened and spot-profile PCCs
are retained as secondary diagnostics because they answer different questions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def per_gene_pcc(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Return one spatial PCC per gene, with NaN for constant targets."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching [spots, genes] tensors")
    pred = prediction.float() - prediction.float().mean(dim=0, keepdim=True)
    truth = target.float() - target.float().mean(dim=0, keepdim=True)
    numerator = torch.sum(pred * truth, dim=0)
    pred_norm = torch.sqrt(torch.sum(pred**2, dim=0))
    target_norm = torch.sqrt(torch.sum(truth**2, dim=0))
    valid_target = target_norm > eps
    result = torch.full_like(numerator, float("nan"))
    valid = valid_target & (pred_norm > eps)
    result[valid] = numerator[valid] / (pred_norm[valid] * target_norm[valid])
    # A constant prediction contains no spatial signal for a variable target.
    result[valid_target & ~valid] = 0.0
    return result


def macro_gene_pcc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average finite gene-wise spatial correlations on one slide."""
    values = per_gene_pcc(prediction, target)
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return prediction.new_tensor(float("nan"))
    return values[finite].mean()


def flattened_pcc(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Correlate all spot-by-gene entries after flattening."""
    pred = prediction.float().reshape(-1)
    truth = target.float().reshape(-1)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denominator = torch.sqrt(torch.sum(pred**2) * torch.sum(truth**2))
    if float(denominator) <= eps:
        return prediction.new_tensor(float("nan"))
    return torch.sum(pred * truth) / denominator


def spot_profile_pcc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average PCC across genes within each spot."""
    values = per_gene_pcc(prediction.T, target.T)
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return prediction.new_tensor(float("nan"))
    return values[finite].mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error over all supplied entries."""
    return torch.sqrt(torch.mean((prediction.float() - target.float()) ** 2))


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all supplied entries."""
    return torch.mean(torch.abs(prediction.float() - target.float()))


def amplitude_ratio(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Median predicted-to-target spatial standard-deviation ratio by gene."""
    pred_std = prediction.float().std(dim=0, correction=0)
    target_std = target.float().std(dim=0, correction=0)
    valid = target_std > 1e-8
    if not bool(valid.any()):
        return prediction.new_tensor(float("nan"))
    return torch.median(pred_std[valid] / target_std[valid])


def effective_rank(
    values: torch.Tensor, max_spots: int = 512, max_genes: int = 2048
) -> torch.Tensor:
    """Entropy effective rank of a centered spot-by-gene matrix."""
    matrix = values.float()
    if len(matrix) > max_spots:
        indices = torch.linspace(0, len(matrix) - 1, max_spots, device=matrix.device).long()
        matrix = matrix[indices]
    if matrix.shape[1] > max_genes:
        indices = torch.linspace(
            0, matrix.shape[1] - 1, max_genes, device=matrix.device
        ).long()
        matrix = matrix[:, indices]
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = energy.sum()
    if float(total) <= 0.0:
        return matrix.new_tensor(0.0)
    probabilities = energy / total
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)))
    return torch.exp(entropy)


def panel_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    panels: Mapping[str, Sequence[int]],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Calculate primary and diagnostic metrics for named gene panels."""
    reports: dict[str, dict[str, float]] = {}
    per_gene: dict[str, np.ndarray] = {}
    for name, raw_indices in panels.items():
        indices = torch.as_tensor(tuple(raw_indices), device=target.device, dtype=torch.long)
        if indices.numel() == 0:
            continue
        pred = prediction[:, indices]
        truth = target[:, indices]
        gene_pcc = per_gene_pcc(pred, truth)
        per_gene[name] = gene_pcc.detach().cpu().numpy()
        finite = torch.isfinite(gene_pcc)
        prediction_rank = (
            float(effective_rank(pred).cpu()) if name == "all_genes" else float("nan")
        )
        target_rank = (
            float(effective_rank(truth).cpu()) if name == "all_genes" else float("nan")
        )
        reports[name] = {
            "gene_pcc": float(gene_pcc[finite].mean().cpu()) if bool(finite.any()) else float("nan"),
            "rmse": float(rmse(pred, truth).cpu()),
            "mae": float(mae(pred, truth).cpu()),
            "spot_profile_pcc": float(spot_profile_pcc(pred, truth).cpu()),
            "flattened_pcc": float(flattened_pcc(pred, truth).cpu()),
            "amplitude_ratio": float(amplitude_ratio(pred, truth).cpu()),
            "prediction_effective_rank": prediction_rank,
            "target_effective_rank": target_rank,
            "n_genes": int(indices.numel()),
            "n_eligible_genes": int(finite.sum().item()),
        }
    return reports, per_gene


def mean_ci(
    values: Sequence[float], *, seed: int = 0, n_bootstrap: int = 2000
) -> tuple[float, float, float]:
    """Return a mean and percentile bootstrap interval across slides."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
