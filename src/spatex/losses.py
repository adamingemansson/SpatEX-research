"""Point-prediction and WAE-MMD training losses.

The point objective combines global flattened PCC with RMSE. The WAE uses an
inverse-multiquadratic MMD term for prior matching.
"""

from __future__ import annotations

import torch


def rmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Compute root mean squared error over all supplied values."""
    mse = torch.mean((prediction - target) ** 2)
    return torch.sqrt(mse + eps)


def pearson_correlation(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute PCC after flattening the supplied spot-by-gene tensors."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = torch.sqrt(
        torch.sum(pred_centered**2) * torch.sum(target_centered**2)
    )
    return torch.sum(pred_centered * target_centered) / denominator.clamp_min(eps)


def point_loss(
    prediction: torch.Tensor, target: torch.Tensor, pcc_weight: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine RMSE with a weighted PCC loss and expose each component."""
    rmse_value = rmse(prediction, target)
    pcc_loss = 1.0 - pearson_correlation(prediction, target)
    total = rmse_value + float(pcc_weight) * pcc_loss
    return total, {"rmse": rmse_value, "pcc_loss": pcc_loss, "total": total}


def imq_mmd(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate maximum mean discrepancy with an inverse multiquadratic kernel."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must be [items, same_features]")
    scale = float(2 * x.shape[1])

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Evaluate the IMQ kernel for every cross-set pair."""
        squared = torch.cdist(a, b) ** 2
        return scale / (scale + squared + eps)

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    if len(x) > 1:
        xx = (k_xx.sum() - k_xx.diagonal().sum()) / (len(x) * (len(x) - 1))
    else:
        xx = k_xx.mean()
    if len(y) > 1:
        yy = (k_yy.sum() - k_yy.diagonal().sum()) / (len(y) * (len(y) - 1))
    else:
        yy = k_yy.mean()
    return xx + yy - 2.0 * k_xy.mean()
