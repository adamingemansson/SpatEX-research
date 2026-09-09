"""Exact deterministic port of the legacy parallel-gated predictor."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from spatex.inputs import InputBatch
from spatex.structure import CenteredGeneStructure


def build_knn_adjacency(coordinates: np.ndarray, k_neighbors: int) -> list[np.ndarray]:
    """Build the legacy symmetric union of directed k-nearest-neighbour edges."""
    n_spots = coordinates.shape[0]
    if n_spots < 2:
        return [np.array([], dtype=int) for _ in range(n_spots)]
    values = np.asarray(coordinates, dtype=np.float64)
    _, nearest = cKDTree(values).query(values, k=min(k_neighbors + 1, n_spots))
    nearest = np.asarray(nearest)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    adjacency = [set() for _ in range(n_spots)]
    for source, row in enumerate(nearest):
        for target in row:
            target = int(target)
            if target != source:
                adjacency[source].add(target)
                adjacency[target].add(source)
    return [np.asarray(sorted(row), dtype=int) for row in adjacency]


def padded_neighbor_graph(
    coordinates: np.ndarray | torch.Tensor,
    k_neighbors: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the legacy ragged neighbour graph for batched attention."""
    if isinstance(coordinates, torch.Tensor):
        values = coordinates.detach().cpu().numpy()
    else:
        values = np.asarray(coordinates)
    adjacency = build_knn_adjacency(values[:, :2], k_neighbors)
    width = max(1, max((len(row) for row in adjacency), default=1))
    indices = np.zeros((len(values), width), dtype=np.int64)
    mask = np.zeros((len(values), width), dtype=bool)
    for row, neighbors in enumerate(adjacency):
        indices[row, : len(neighbors)] = neighbors
        mask[row, : len(neighbors)] = True
    return (
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


class WeightedGeneExpressionEncoder(nn.Module):
    """Legacy bias-free weighted gene projection."""

    def __init__(self, n_genes: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(n_genes, output_dim, bias=False)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.projection(expression)


class FourierCoordinateEncoding(nn.Module):
    """Legacy Fourier encoding of normalized two-dimensional coordinates."""

    def __init__(self, output_dim: int = 64, num_frequencies: int = 8) -> None:
        super().__init__()
        frequencies = (2.0 ** torch.arange(num_frequencies)) * math.pi
        self.register_buffer("freqs", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(4 * num_frequencies, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = (coordinates[..., None] * self.freqs).reshape(
            *coordinates.shape[:-1], -1
        )
        return self.mlp(torch.cat((projected.sin(), projected.cos()), dim=-1))


class SpotTokenProjection(nn.Module):
    """Legacy five-branch spot-token projection."""

    def __init__(
        self,
        hidden_dim: int,
        image_feature_dim: int,
        image_proj_dim: int,
        gex_feature_dim: int,
        gex_proj_dim: int,
        coord_dim: int,
        n_modality_flags: int,
        modality_flag_dim: int,
        n_boundary_rings: int = 4,
        ring_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.image_norm = nn.LayerNorm(image_feature_dim)
        self.image_proj = nn.Linear(image_feature_dim, image_proj_dim)
        self.gex_norm = nn.LayerNorm(gex_feature_dim)
        self.gex_proj = nn.Linear(gex_feature_dim, gex_proj_dim)
        self.coord_encoding = FourierCoordinateEncoding(output_dim=coord_dim)
        self.ring_embedding = nn.Embedding(n_boundary_rings, ring_embed_dim)
        self.modality_flag_proj = nn.Linear(n_modality_flags, modality_flag_dim)
        width = image_proj_dim + gex_proj_dim + coord_dim + ring_embed_dim + modality_flag_dim
        self.output_proj = nn.Linear(width, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.n_boundary_rings = int(n_boundary_rings)

    def forward(
        self,
        image_features: torch.Tensor,
        gex_features: torch.Tensor,
        coords: torch.Tensor,
        boundary_ring: torch.Tensor,
        modality_flags: torch.Tensor,
    ) -> torch.Tensor:
        parts = (
            self.image_proj(self.image_norm(image_features)),
            self.gex_proj(self.gex_norm(gex_features)),
            self.coord_encoding(coords),
            self.ring_embedding(boundary_ring),
            self.modality_flag_proj(modality_flags),
        )
        return self.output_norm(self.output_proj(torch.cat(parts, dim=-1)))


class RelativeGeometryBias(nn.Module):
    """Legacy relative-position bias for every attention head."""

    def __init__(self, n_heads: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_heads)
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.mlp(geometry)


class _CachedGeometrySelfAttention(nn.Module):
    """Legacy dense/sparse geometry-aware self-attention."""

    def __init__(self, hidden_dim: int, n_heads: int, *, dense_threshold: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = hidden_dim // n_heads
        self.dense_threshold = int(dense_threshold)
        self.last_attention_mode: str | None = None
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_bias = RelativeGeometryBias(n_heads)

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_spots, n_neighbors = neighbor_indices.shape
        if n_spots <= self.dense_threshold:
            neighbor_indices = torch.arange(n_spots, device=hidden.device)[None, :].expand(n_spots, -1)
            neighbor_mask = torch.ones((n_spots, n_spots), dtype=torch.bool, device=hidden.device)
            n_neighbors = n_spots
            self.last_attention_mode = "dense"
        else:
            self.last_attention_mode = "sparse"
        query = self.query_proj(hidden).view(n_spots, self.n_heads, self.head_dim)
        key = self.key_proj(hidden).view(n_spots, self.n_heads, self.head_dim)
        value = self.value_proj(hidden).view(n_spots, self.n_heads, self.head_dim)
        rows = torch.arange(n_spots, device=hidden.device)[:, None].expand(-1, n_neighbors)
        offsets = coords[neighbor_indices] - coords[rows]
        distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
        bias = self.geometry_bias(torch.cat((offsets, distance), dim=-1)).permute(0, 2, 1)
        logits = torch.einsum("nhd,nkhd->nhk", query, key[neighbor_indices])
        logits = logits / math.sqrt(self.head_dim)
        logits = (logits + bias).masked_fill(~neighbor_mask[:, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        output = torch.einsum("nhk,nkhd->nhd", weights, value[neighbor_indices])
        return self.out_proj(output.reshape(n_spots, self.hidden_dim))


class _ImageSpatialBlock(nn.Module):
    """Legacy pre-normalized spatial attention block."""

    def __init__(
        self, hidden_dim: int, n_heads: int, dense_threshold: int, sparse_k: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.sparse_k = int(sparse_k)
        self.cached_attention = _CachedGeometrySelfAttention(
            hidden_dim, n_heads, dense_threshold=dense_threshold
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        update = self.cached_attention(
            self.norm_attention(hidden), coords, neighbor_indices, neighbor_mask
        )
        hidden = hidden + update
        return hidden + self.ffn(self.norm_ffn(hidden))


class Architecture1ImageConditioner(nn.Module):
    """Exact Task-I image conditioner from the legacy implementation."""

    def __init__(
        self,
        n_genes: int,
        image_feature_dim: int = 1536,
        gex_feature_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        coord_dim: int = 64,
        image_proj_dim: int = 256,
        gex_proj_dim: int = 256,
        modality_flag_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_feature_dim = int(image_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_genes = int(n_genes)
        self.gex_feature_dim = int(gex_feature_dim)
        # This unused Task-I branch existed in the original and affects initialization order.
        self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, gex_feature_dim)
        self.spot_token = SpotTokenProjection(
            hidden_dim,
            image_feature_dim,
            image_proj_dim,
            gex_feature_dim,
            gex_proj_dim,
            coord_dim,
            2,
            modality_flag_dim,
        )
        self.blocks = nn.ModuleList(
            _ImageSpatialBlock(hidden_dim, n_heads, dense_threshold, sparse_k, dropout)
            for _ in range(n_blocks)
        )
        self.sparse_k = int(sparse_k)
        self._neighbor_graph_cache: dict[tuple[int, bytes], tuple[torch.Tensor, torch.Tensor]] = {}

    def _graph(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.ascontiguousarray(coordinates.detach().cpu().numpy().astype(np.float64))
        key = (len(values), hashlib.blake2b(values.tobytes(), digest_size=16).digest())
        cached = self._neighbor_graph_cache.get(key)
        if cached is None:
            cached = padded_neighbor_graph(values, self.sparse_k)
            if len(self._neighbor_graph_cache) >= 512:
                self._neighbor_graph_cache.pop(next(iter(self._neighbor_graph_cache)))
            self._neighbor_graph_cache[key] = cached
        return cached[0].to(coordinates.device), cached[1].to(coordinates.device)

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        inputs.validate(self.image_feature_dim)
        image = inputs.image_features.float()
        coords = inputs.coordinates.float()
        available = inputs.image_available.float().unsqueeze(-1)
        gex_features = torch.zeros(
            (len(image), self.gex_feature_dim), dtype=image.dtype, device=image.device
        )
        expression_available = torch.zeros_like(available)
        hidden = self.spot_token(
            image,
            gex_features,
            coords,
            torch.zeros(len(image), dtype=torch.long, device=image.device),
            torch.cat((available, expression_available), dim=-1),
        )
        neighbor_indices, neighbor_mask = self._graph(coords)
        for block in self.blocks:
            hidden = block(hidden, coords, neighbor_indices, neighbor_mask)
        return hidden[inputs.query_mask]


class CenteredGeneStructureRefinement(nn.Module):
    """Exact legacy residual update in the fixed centered gene basis."""

    def __init__(self, structure: CenteredGeneStructure, hidden_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("basis_matrix", structure.basis.clone(), persistent=False)
        self.register_buffer("global_gene_mean", structure.global_mean.clone(), persistent=False)
        self.register_buffer("per_gene_scale", structure.per_gene_scale.clone(), persistent=False)
        rank = structure.basis.shape[0]
        self.refine = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        standardized = (prediction - self.global_gene_mean) / self.per_gene_scale
        coefficients = standardized @ self.basis_matrix.T
        delta = self.refine(coefficients) @ self.basis_matrix
        return prediction + delta * self.per_gene_scale


class SpatialExpressionRefiner(nn.Module):
    """Exact legacy between-spot expression refiner."""

    def __init__(
        self,
        n_genes: int,
        context_dim: int,
        *,
        gex_feature_dim: int = 256,
        hidden_dim: int = 256,
        geometry_dim: int = 32,
        k_neighbors: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.k_neighbors = int(k_neighbors)
        self._neighbor_graph_cache: dict[tuple[int, bytes], tuple[torch.Tensor, torch.Tensor]] = {}
        self._neighbor_graph_cache_size = 512
        self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, gex_feature_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.expression_proj = nn.Linear(gex_feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(3, geometry_dim), nn.GELU(), nn.Linear(geometry_dim, geometry_dim)
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + geometry_dim + gex_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.update_head[-1].weight)
        nn.init.zeros_(self.update_head[-1].bias)

    def cached_neighbor_graph(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.ascontiguousarray(coordinates.detach().cpu().numpy()[:, :2].astype(np.float64))
        key = (len(values), hashlib.blake2b(values.tobytes(), digest_size=16).digest())
        cached = self._neighbor_graph_cache.get(key)
        if cached is None:
            cached = padded_neighbor_graph(values, self.k_neighbors)
            if len(self._neighbor_graph_cache) >= self._neighbor_graph_cache_size:
                self._neighbor_graph_cache.pop(next(iter(self._neighbor_graph_cache)))
            self._neighbor_graph_cache[key] = cached
        return cached[0].to(coordinates.device), cached[1].to(coordinates.device)

    def forward(
        self,
        expression: torch.Tensor,
        context: torch.Tensor,
        coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        genes = self.gene_encoder(expression)
        hidden = self.norm(self.context_proj(context) + self.expression_proj(genes))
        gathered_hidden = hidden[neighbor_indices]
        gathered_genes = genes[neighbor_indices]
        offsets = coords[neighbor_indices][..., :2] - coords[:, None, :2]
        distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
        geometry = self.geometry_mlp(torch.cat((offsets, distance), dim=-1))
        pairs = torch.cat(
            (
                hidden[:, None, :].expand_as(gathered_hidden),
                gathered_hidden,
                geometry,
                genes[:, None, :] - gathered_genes,
            ),
            dim=-1,
        )
        logits = self.attention_mlp(pairs).squeeze(-1)
        logits = logits.masked_fill(~neighbor_mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        aggregate = torch.einsum("nk,nkh->nh", weights, self.value_proj(gathered_hidden))
        return expression + self.update_head(hidden + aggregate)


class LegacyParallelGatedSpatEX(nn.Module):
    """Layer-for-layer port of the legacy deterministic parallel-gated model."""

    def __init__(self, config: dict, structure: CenteredGeneStructure) -> None:
        super().__init__()
        model = config["model"]
        n_genes = len(structure.gene_names)
        context_dim = int(model["context_dim"])
        self.gene_names = structure.gene_names
        self.n_genes = n_genes
        self.image_conditioner = Architecture1ImageConditioner(
            n_genes=n_genes,
            image_feature_dim=int(model["image_feature_dim"]),
            gex_feature_dim=int(model.get("gex_proj_dim", 256)),
            hidden_dim=context_dim,
            n_heads=int(model["n_heads"]),
            n_blocks=int(model["n_blocks"]),
            dense_threshold=int(model["dense_threshold"]),
            sparse_k=int(model["spatial_k"]),
            coord_dim=int(model["coord_dim"]),
            image_proj_dim=int(model["image_proj_dim"]),
            gex_proj_dim=int(model.get("gex_proj_dim", 256)),
            modality_flag_dim=int(model.get("modality_flag_dim", 16)),
            dropout=float(model["dropout"]),
        )
        hidden_dim = int(model["decoder_hidden_dim"])
        self.conditional_mean_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.coexpression_refinement = CenteredGeneStructureRefinement(
            structure, hidden_dim=int(model["gene_structure_hidden_dim"])
        )
        self.n_refinement_steps = int(model.get("refinement_steps", 1))
        self.structured_composition = "parallel_gated"
        self.spatial_refiner = SpatialExpressionRefiner(
            n_genes,
            context_dim,
            gex_feature_dim=int(model["refinement_gex_dim"]),
            hidden_dim=int(model["refinement_hidden_dim"]),
            k_neighbors=int(model["refinement_k"]),
        )
        gate_logit = math.log(0.1 / 0.9)
        self.composition_gate_logits = nn.Parameter(
            torch.full((2,), gate_logit, dtype=torch.float32)
        )
        self.register_buffer("per_gene_scale", structure.per_gene_scale.clone(), persistent=True)

    def encode(self, inputs: InputBatch) -> torch.Tensor:
        """Return legacy image context for query spots only."""
        return self.image_conditioner(inputs)

    def _apply_between(
        self, expression: torch.Tensor, context: torch.Tensor, inputs: InputBatch
    ) -> torch.Tensor:
        coordinates = inputs.coordinates[inputs.query_mask].to(expression)
        neighbor_indices, neighbor_mask = self.spatial_refiner.cached_neighbor_graph(coordinates)
        for _ in range(self.n_refinement_steps):
            expression = self.spatial_refiner(
                expression, context, coordinates, neighbor_indices, neighbor_mask
            )
        return expression

    def predict_all(self, inputs: InputBatch) -> dict[str, torch.Tensor]:
        """Return the exact legacy prediction and both refinement paths."""
        context = self.encode(inputs)
        base = self.conditional_mean_head(context)
        within = self.coexpression_refinement(base)
        between = self._apply_between(base, context, inputs)
        gates = torch.sigmoid(self.composition_gate_logits)
        expression = base + gates[0] * (within - base) + gates[1] * (between - base)
        return {
            "expression": expression,
            "context": context,
            "base": base,
            "within": within,
            "between": between,
            "gates": gates,
        }

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        """Return expression for query spots."""
        return self.predict_all(inputs)["expression"]
