"""Define the expression-free input contract used at deployment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InputBatch:
    """Inputs available during H&E-only inference."""

    sample_id: str
    image_features: torch.Tensor
    coordinates: torch.Tensor
    image_available: torch.Tensor
    query_mask: torch.Tensor

    def validate(self, image_feature_dim: int | None = None) -> "InputBatch":
        """Check tensor shapes, types, values, and the query selection."""
        # Keeping expression out of this class prevents accidental GEX leakage.
        if self.image_features.ndim != 2:
            raise ValueError("image_features must be [spots, features]")
        n_spots, feature_dim = self.image_features.shape
        if image_feature_dim is not None and feature_dim != image_feature_dim:
            raise ValueError(
                f"expected image feature dimension {image_feature_dim}, got {feature_dim}"
            )
        if self.coordinates.shape != (n_spots, 2):
            raise ValueError("coordinates must be [spots, 2]")
        if self.image_available.shape != (n_spots,):
            raise ValueError("image_available must be [spots]")
        if self.query_mask.shape != (n_spots,):
            raise ValueError("query_mask must be [spots]")
        if self.image_available.dtype != torch.bool or self.query_mask.dtype != torch.bool:
            raise ValueError("availability and query masks must be boolean")
        if not bool(self.query_mask.any()):
            raise ValueError("at least one query spot is required")
        for name, tensor in (
            ("image_features", self.image_features),
            ("coordinates", self.coordinates),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains non-finite values")
        return self

    def to(self, device: torch.device | str) -> "InputBatch":
        """Return the same input batch on another device."""
        return InputBatch(
            sample_id=self.sample_id,
            image_features=self.image_features.to(device),
            coordinates=self.coordinates.to(device),
            image_available=self.image_available.to(device),
            query_mask=self.query_mask.to(device),
        )

    @property
    def n_queries(self) -> int:
        """Number of spots for which expression will be returned."""
        return int(self.query_mask.sum().item())
