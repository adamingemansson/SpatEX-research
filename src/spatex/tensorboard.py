"""Compact TensorBoard figures for validation and WAE diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _new_figure(*, figsize: tuple[float, float]):
    """Create an Agg-backed figure only when visual logging is enabled."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def spatial_prediction_figure(
    coordinates: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_id: str,
    gene: str,
):
    """Plot target, prediction, and absolute error on the same spot grid."""
    coords = coordinates.detach().float().cpu().numpy()
    truth = target.detach().float().cpu().numpy()
    estimate = prediction.detach().float().cpu().numpy()
    error = np.abs(estimate - truth)
    finite = np.isfinite(truth) & np.isfinite(estimate)
    if not bool(finite.any()):
        raise ValueError("spatial diagnostic has no finite values")

    shared = np.concatenate((truth[finite], estimate[finite]))
    lower, upper = np.nanpercentile(shared, (1.0, 99.0))
    lower = min(0.0, float(lower))
    upper = max(float(upper), lower + 1e-6)
    error_upper = max(float(np.nanpercentile(error[finite], 99.0)), 1e-6)

    centered_truth = truth[finite] - float(np.mean(truth[finite]))
    centered_estimate = estimate[finite] - float(np.mean(estimate[finite]))
    denominator = float(
        np.sqrt(np.sum(centered_truth**2) * np.sum(centered_estimate**2))
    )
    pcc = (
        float(np.sum(centered_truth * centered_estimate) / denominator)
        if denominator
        else float("nan")
    )
    gene_rmse = float(np.sqrt(np.mean((estimate[finite] - truth[finite]) ** 2)))

    figure = _new_figure(figsize=(12, 4))
    axes = figure.subplots(1, 3)
    panels = (
        ("Target", truth, "viridis", lower, upper),
        ("Prediction", estimate, "viridis", lower, upper),
        ("Absolute error", error, "magma", 0.0, error_upper),
    )
    point_size = max(2.0, min(12.0, 18000.0 / max(len(coords), 1)))
    for axis, (title, values, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        scatter = axis.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        f"{sample_id} | {gene} | PCC={pcc:.3f} | RMSE={gene_rmse:.3f}"
    )
    return figure


def latent_pca_figure(
    latent: torch.Tensor,
    labels: Sequence[str],
    *,
    max_points: int,
):
    """Project posterior latents to two dimensions and color them by organ."""
    values = latent.detach().float().cpu()
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("latent PCA inputs have incompatible shapes")
    if len(values) < 2:
        raise ValueError("latent PCA requires at least two points")
    if len(values) > max_points:
        indices = torch.linspace(0, len(values) - 1, max_points).long()
        values = values[indices]
        labels = [labels[int(index)] for index in indices]
    values = values - values.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(values, q=2, center=False)
    projected = (values @ vectors[:, :2]).numpy()

    figure = _new_figure(figsize=(6, 5))
    axis = figure.subplots()
    labels_array = np.asarray(labels, dtype=str)
    for organ in sorted(set(labels_array.tolist())):
        selected = labels_array == organ
        axis.scatter(
            projected[selected, 0],
            projected[selected, 1],
            s=5,
            alpha=0.65,
            linewidths=0,
            label=organ,
        )
    axis.set_title("Posterior latent z: PCA")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(loc="best", fontsize=7, frameon=False)
    return figure


def example_map(logging: dict, gene_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Resolve configured slide/gene examples to gene indices."""
    lookup = {name: index for index, name in enumerate(gene_names)}
    resolved: dict[str, list[int]] = {}
    for example in logging.get("spatial_examples", []):
        sample_id = str(example["sample_id"])
        genes = [str(value) for value in example.get("genes", [])]
        indices = [lookup[gene] for gene in genes if gene in lookup]
        if indices:
            resolved.setdefault(sample_id, []).extend(indices)
    return {sample: tuple(dict.fromkeys(indices)) for sample, indices in resolved.items()}
