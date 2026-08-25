"""Coordinate normalization and spatial-neighbor utilities."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Center coordinates and scale them by median nearest-spot spacing."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be [spots, 2]")
    if len(coordinates) < 2:
        return coordinates - coordinates.mean(axis=0, keepdims=True)
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("could not determine positive spot spacing")
    return ((coordinates - coordinates.mean(axis=0, keepdims=True)) / spacing).astype(
        np.float32
    )


def knn_indices(coordinates: torch.Tensor, k: int) -> torch.Tensor:
    """Return the nearest spot indices for each spot, including itself."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be [spots, 2]")
    n = coordinates.shape[0]
    if n == 0:
        raise ValueError("empty coordinate set")
    k_eff = min(max(int(k), 1), n)
    # Self is retained as a stable reference when aggregating neighbors.
    distances = torch.cdist(coordinates.float(), coordinates.float())
    return torch.topk(distances, k=k_eff, largest=False).indices


def neighbor_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a dense neighbor axis from a spot-level tensor."""
    if values.ndim != 2 or indices.ndim != 2:
        raise ValueError("values and indices must be rank two")
    return values[indices]
