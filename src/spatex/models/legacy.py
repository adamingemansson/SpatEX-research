"""Legacy modules used by the original parallel-gated SpatEX model."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from spatex.inputs import InputBatch


def _symmetric_knn(
    coordinates: torch.Tensor, k_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the padded symmetric k-NN union without self-loops."""
    values = np.asarray(
        coordinates.detach().cpu().numpy(), dtype=np.float64, order="C"
    )
    n_spots = len(values)
    if n_spots < 2:
        return (
            torch.zeros((n_spots, 1), dtype=torch.long, device=coordinates.device),
            torch.zeros((n_spots, 1), dtype=torch.bool, device=coordinates.device),
        )
    width = min(int(k_neighbors) + 1, n_spots)
    _, nearest = cKDTree(values).query(values, k=width)
    nearest = np.asarray(nearest)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    adjacency = [set() for _ in range(n_spots)]
    for source, row in enumerate(nearest):
        for target in row:
            target = int(target)
            if target == source:
                continue
            adjacency[source].add(target)
            adjacency[target].add(source)
    padded_width = max(1, max(map(len, adjacency)))
    indices = torch.zeros(
        (n_spots, padded_width), dtype=torch.long, device=coordinates.device
    )
    mask = torch.zeros(
        (n_spots, padded_width), dtype=torch.bool, device=coordinates.device
    )
    for source, neighbors in enumerate(adjacency):
        ordered = sorted(neighbors)
        if ordered:
            count = len(ordered)
            indices[source, :count] = torch.as_tensor(
                ordered, dtype=torch.long, device=coordinates.device
            )
            mask[source, :count] = True
    return indices, mask


class LegacyFourierCoordinates(nn.Module):
    """Encode normalized spot coordinates as in the original model."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        frequencies = (2.0 ** torch.arange(8)) * math.pi
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(32, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = (coordinates[..., None] * self.frequencies).reshape(
            len(coordinates), -1
        )
        return self.mlp(torch.cat((projected.sin(), projected.cos()), dim=-1))


class LegacyGeometryAttention(nn.Module):
    """Original dense/sparse geometry-aware self-attention."""

    def __init__(self, dimension: int, n_heads: int, dense_threshold: int) -> None:
        super().__init__()
        if dimension % n_heads:
            raise ValueError("context dimension must be divisible by n_heads")
        self.dimension = int(dimension)
        self.n_heads = int(n_heads)
        self.head_dim = self.dimension // self.n_heads
        self.dense_threshold = int(dense_threshold)
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.geometry_bias = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, n_heads)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_spots = len(hidden)
        query = self.query(hidden).view(n_spots, self.n_heads, self.head_dim)
        key = self.key(hidden).view(n_spots, self.n_heads, self.head_dim)
        value = self.value(hidden).view(n_spots, self.n_heads, self.head_dim)
        if n_spots <= self.dense_threshold:
            neighbor_indices = torch.arange(
                n_spots, device=hidden.device, dtype=torch.long
            )[None, :].expand(n_spots, -1)
            neighbor_mask = torch.ones(
                (n_spots, n_spots), device=hidden.device, dtype=torch.bool
            )
        offsets = coordinates[neighbor_indices] - coordinates[:, None, :]
        distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
        bias = self.geometry_bias(torch.cat((offsets, distance), dim=-1)).permute(
            0, 2, 1
        )
        gathered_key = key[neighbor_indices]
        gathered_value = value[neighbor_indices]
        logits = torch.einsum("nhd,nkhd->nhk", query, gathered_key)
        logits = logits / math.sqrt(self.head_dim)
        logits = (logits + bias).masked_fill(
            ~neighbor_mask[:, None, :], float("-inf")
        )
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        attended = torch.einsum("nhk,nkhd->nhd", weights, gathered_value)
        return self.output(attended.reshape(n_spots, self.dimension))


class LegacySpatialBlock(nn.Module):
    """Original pre-norm attention and feed-forward residual block."""

    def __init__(
        self, dimension: int, n_heads: int, dense_threshold: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = LegacyGeometryAttention(
            dimension, n_heads, dense_threshold
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            coordinates,
            neighbor_indices,
            neighbor_mask,
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


class LegacyImageSpatialConditioner(nn.Module):
    """Original H&E-only token projection and spatial conditioner."""

    def __init__(
        self,
        image_feature_dim: int,
        image_proj_dim: int,
        gex_proj_dim: int,
        coord_dim: int,
        context_dim: int,
        n_heads: int,
        n_blocks: int,
        dense_threshold: int,
        spatial_k: int,
        ring_embed_dim: int,
        modality_flag_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.image_feature_dim = int(image_feature_dim)
        self.spatial_k = int(spatial_k)
        self.image_norm = nn.LayerNorm(image_feature_dim)
        self.image_projection = nn.Linear(image_feature_dim, image_proj_dim)
        self.gex_norm = nn.LayerNorm(gex_proj_dim)
        self.gex_projection = nn.Linear(gex_proj_dim, gex_proj_dim)
        self.coordinates = LegacyFourierCoordinates(coord_dim)
        self.ring_embedding = nn.Embedding(4, ring_embed_dim)
        self.modality_projection = nn.Linear(2, modality_flag_dim)
        concatenated = (
            image_proj_dim
            + gex_proj_dim
            + coord_dim
            + ring_embed_dim
            + modality_flag_dim
        )
        self.input_projection = nn.Linear(concatenated, context_dim)
        self.input_norm = nn.LayerNorm(context_dim)
        self.blocks = nn.ModuleList(
            LegacySpatialBlock(context_dim, n_heads, dense_threshold, dropout)
            for _ in range(n_blocks)
        )
        self._graph_cache: dict[tuple[int, bytes], tuple[torch.Tensor, torch.Tensor]] = {}

    def _graph(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.asarray(
            coordinates.detach().cpu().numpy(), dtype=np.float64, order="C"
        )
        key = (len(values), hashlib.blake2b(values.tobytes(), digest_size=16).digest())
        if key not in self._graph_cache:
            if len(self._graph_cache) >= 512:
                self._graph_cache.pop(next(iter(self._graph_cache)))
            indices, mask = _symmetric_knn(coordinates, self.spatial_k)
            self._graph_cache[key] = (indices.cpu(), mask.cpu())
        indices, mask = self._graph_cache[key]
        return indices.to(coordinates.device), mask.to(coordinates.device)

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        inputs.validate(self.image_feature_dim)
        available = inputs.image_available[:, None].to(inputs.image_features.dtype)
        image = inputs.image_features * available
        image = self.image_projection(self.image_norm(image))
        gex = torch.zeros(
            (len(image), self.gex_norm.normalized_shape[0]),
            device=image.device,
            dtype=image.dtype,
        )
        gex = self.gex_projection(self.gex_norm(gex))
        coordinates = self.coordinates(inputs.coordinates)
        rings = self.ring_embedding(
            torch.zeros(len(image), device=image.device, dtype=torch.long)
        )
        modality = self.modality_projection(
            torch.cat((available, torch.zeros_like(available)), dim=-1)
        )
        hidden = self.input_norm(
            self.input_projection(
                torch.cat((image, gex, coordinates, rings, modality), dim=-1)
            )
        )
        indices, mask = self._graph(inputs.coordinates)
        for block in self.blocks:
            hidden = block(hidden, inputs.coordinates, indices, mask)
        return hidden


class LegacyBetweenSpotRefiner(nn.Module):
    """Original between-spot GEX attention and residual update."""

    def __init__(
        self,
        n_genes: int,
        context_dim: int,
        gex_dim: int,
        hidden_dim: int,
        k_neighbors: int,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self._graph_cache: dict[tuple[int, bytes], tuple[torch.Tensor, torch.Tensor]] = {}
        self.gene_encoder = nn.Linear(n_genes, gex_dim, bias=False)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.expression_projection = nn.Linear(gex_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.geometry = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + 32 + gex_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def _graph(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.asarray(
            coordinates.detach().cpu().numpy(), dtype=np.float64, order="C"
        )
        key = (len(values), hashlib.blake2b(values.tobytes(), digest_size=16).digest())
        if key not in self._graph_cache:
            if len(self._graph_cache) >= 512:
                self._graph_cache.pop(next(iter(self._graph_cache)))
            indices, mask = _symmetric_knn(coordinates, self.k_neighbors)
            self._graph_cache[key] = (indices.cpu(), mask.cpu())
        indices, mask = self._graph_cache[key]
        return indices.to(coordinates.device), mask.to(coordinates.device)

    def forward(
        self, expression: torch.Tensor, context: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        indices, mask = self._graph(coordinates)
        genes = self.gene_encoder(expression)
        hidden = self.hidden_norm(
            self.context_projection(context) + self.expression_projection(genes)
        )
        neighbor_hidden = hidden[indices]
        neighbor_genes = genes[indices]
        offsets = coordinates[indices] - coordinates[:, None, :]
        distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
        geometry = self.geometry(torch.cat((offsets, distance), dim=-1))
        pairs = torch.cat(
            (
                hidden[:, None, :].expand_as(neighbor_hidden),
                neighbor_hidden,
                geometry,
                genes[:, None, :] - neighbor_genes,
            ),
            dim=-1,
        )
        logits = self.attention(pairs).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        aggregate = torch.einsum(
            "nk,nkh->nh", weights, self.value(neighbor_hidden)
        )
        return expression + self.update(hidden + aggregate)
