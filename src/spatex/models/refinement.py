"""Refine expression within each spot and between neighboring spots.

The within branch acts in a learned gene-program basis. The between branch
attends over nearby predicted profiles and image context.
"""

from __future__ import annotations

import torch
from torch import nn

from spatex.graph import knn_indices, neighbor_gather
from spatex.structure import CenteredGeneStructure


class WithinGeneRefiner(nn.Module):
    """Refine co-expression along learned gene programs."""

    def __init__(self, structure: CenteredGeneStructure, hidden_dim: int) -> None:
        super().__init__()
        structure.validate()
        self.register_buffer("basis", structure.basis.clone())
        self.register_buffer("global_mean", structure.global_mean.clone())
        self.register_buffer("per_gene_scale", structure.per_gene_scale.clone())
        rank = structure.basis.shape[0]
        self.program_network = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )
        # Start as an identity map and learn only the correction.
        nn.init.zeros_(self.program_network[-1].weight)
        nn.init.zeros_(self.program_network[-1].bias)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """Apply a residual update in the fitted gene-program space."""
        standardized = (expression - self.global_mean) / self.per_gene_scale
        programs = standardized @ self.basis.T
        delta = self.program_network(programs) @ self.basis
        return expression + delta * self.per_gene_scale


class BetweenSpotRefiner(nn.Module):
    """Refine each spot from nearby predicted expression and image context."""

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
        self.gene_encoder = nn.Sequential(nn.Linear(n_genes, gex_dim), nn.GELU())
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
        # Start as an identity map and learn only the correction.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(
        self, expression: torch.Tensor, context: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Attend to neighboring predicted profiles and add a spatial update."""
        indices = knn_indices(coordinates, self.k_neighbors)
        gene_features = self.gene_encoder(expression)
        hidden = self.hidden_norm(
            self.context_projection(context) + self.expression_projection(gene_features)
        )
        neighbor_hidden = neighbor_gather(hidden, indices)
        neighbor_genes = neighbor_gather(gene_features, indices)
        offsets = neighbor_gather(coordinates, indices) - coordinates[:, None, :]
        distance = torch.linalg.norm(offsets, dim=-1, keepdim=True)
        geometry = self.geometry(torch.cat((offsets, distance), dim=-1))
        # Attention compares image context, geometry, and predicted GEX differences.
        pair = torch.cat(
            (
                hidden[:, None, :].expand_as(neighbor_hidden),
                neighbor_hidden,
                geometry,
                gene_features[:, None, :] - neighbor_genes,
            ),
            dim=-1,
        )
        scores = self.attention(pair).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        aggregate = torch.sum(weights[..., None] * self.value(neighbor_hidden), dim=1)
        return expression + self.update(hidden + aggregate)
