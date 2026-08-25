"""Encode UNI2 features and spatial geometry into spot-level context.

Small fields use dense attention; whole slides use local spatial attention to
keep memory bounded.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spatex.graph import knn_indices, neighbor_gather
from spatex.inputs import InputBatch


class FourierCoordinates(nn.Module):
    """Map normalized two-dimensional coordinates to learned Fourier features."""
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        frequencies = (2.0 ** torch.arange(8)) * math.pi
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(32, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode one normalized coordinate pair per spot."""
        phase = coordinates[..., None] * self.frequencies
        features = torch.cat(
            (
                torch.sin(phase[:, 0]),
                torch.cos(phase[:, 0]),
                torch.sin(phase[:, 1]),
                torch.cos(phase[:, 1]),
            ),
            dim=-1,
        )
        return self.mlp(features)


class GeometryAttentionBlock(nn.Module):
    """Self-attention with learned relative-position bias."""
    def __init__(
        self,
        dimension: int,
        n_heads: int,
        dropout: float,
        dense_threshold: int,
        spatial_k: int,
    ) -> None:
        super().__init__()
        if dimension % n_heads:
            raise ValueError("context dimension must be divisible by n_heads")
        self.dimension = dimension
        self.n_heads = n_heads
        self.head_dim = dimension // n_heads
        self.norm1 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.geometry_bias = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, n_heads)
        )
        self.dense_threshold = int(dense_threshold)
        self.spatial_k = int(spatial_k)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Exchange context globally for small fields and locally for slides."""
        n = len(values)
        normalized = self.norm1(values)
        qkv = self.qkv(normalized).reshape(n, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        if n <= self.dense_threshold:
            # Small fields can attend to every spot.
            scores = torch.einsum("ihd,jhd->hij", query, key) / math.sqrt(self.head_dim)
            delta = coordinates[None, :, :] - coordinates[:, None, :]
            distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
            bias = self.geometry_bias(torch.cat((delta, distance), dim=-1)).permute(2, 0, 1)
            scores = scores + bias
            weights = torch.softmax(scores, dim=-1)
            attended = torch.einsum("hij,jhd->ihd", weights, value)
        else:
            # Whole slides use local attention to keep memory bounded.
            indices = knn_indices(coordinates, self.spatial_k)
            keys = key[indices]
            neighbor_values = value[indices]
            offsets = coordinates[:, None, :] - coordinates[indices]
            distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
            scores = torch.einsum("ihd,ikhd->ihk", query, keys) / math.sqrt(self.head_dim)
            bias = self.geometry_bias(torch.cat((offsets, distance), dim=-1)).permute(0, 2, 1)
            scores = scores + bias
            weights = torch.softmax(scores, dim=-1)
            attended = torch.einsum("ihk,ikhd->ihd", weights, neighbor_values)
        attended = attended.reshape(n, self.dimension)
        values = values + self.dropout(self.output(attended))
        return values + self.mlp(self.norm2(values))


class ImageSpatialConditioner(nn.Module):
    """Build one spatial context vector per spot."""

    def __init__(
        self,
        image_feature_dim: int,
        image_proj_dim: int,
        coord_dim: int,
        context_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        dense_threshold: int,
        spatial_k: int,
        gex_proj_dim: int = 256,
        ring_embed_dim: int = 16,
        modality_flag_dim: int = 16,
    ) -> None:
        super().__init__()
        self.image_feature_dim = int(image_feature_dim)
        self.image_projection = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, image_proj_dim),
            nn.GELU(),
        )
        self.zero_gex_projection = nn.Sequential(
            nn.LayerNorm(gex_proj_dim), nn.Linear(gex_proj_dim, gex_proj_dim)
        )
        self.coordinates = FourierCoordinates(coord_dim)
        self.ring_embedding = nn.Embedding(4, ring_embed_dim)
        self.modality_projection = nn.Linear(2, modality_flag_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(
                image_proj_dim
                + gex_proj_dim
                + coord_dim
                + ring_embed_dim
                + modality_flag_dim,
                context_dim,
            ),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            GeometryAttentionBlock(
                context_dim, n_heads, dropout, dense_threshold, spatial_k
            )
            for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(context_dim)

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        """Fuse image, coordinate, availability, and fixed modality tokens."""
        inputs.validate(self.image_feature_dim)
        # Missing image patches contribute zeros plus an availability flag.
        image = inputs.image_features * inputs.image_available[:, None].to(
            inputs.image_features.dtype
        )
        encoded = self.image_projection(image)
        coordinate_features = self.coordinates(inputs.coordinates)
        available = inputs.image_available[:, None].to(encoded.dtype)
        zero_gex = torch.zeros(
            len(encoded), self.zero_gex_projection[0].normalized_shape[0],
            device=encoded.device, dtype=encoded.dtype,
        )
        gex = self.zero_gex_projection(zero_gex)
        # H&E-only inference uses the fixed no-GEX and centre-ring tokens.
        rings = self.ring_embedding(
            torch.zeros(len(encoded), device=encoded.device, dtype=torch.long)
        )
        modality = self.modality_projection(
            torch.cat((available, torch.zeros_like(available)), dim=-1)
        )
        context = self.input_projection(
            torch.cat((encoded, gex, coordinate_features, rings, modality), dim=-1)
        )
        for block in self.blocks:
            context = block(context, inputs.coordinates)
        return self.final_norm(context)
