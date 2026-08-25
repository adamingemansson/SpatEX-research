"""Define the deterministic SpatEX predictor.

A full-GEX decoder is followed by parallel within-gene and between-spot
residual refiners whose contributions are learned through two gates.
"""

from __future__ import annotations

import torch
from torch import nn

from spatex.inputs import InputBatch
from spatex.models.conditioner import ImageSpatialConditioner
from spatex.models.refinement import BetweenSpotRefiner, WithinGeneRefiner
from spatex.structure import CenteredGeneStructure


class SpatEX(nn.Module):
    """Predict full spatial expression from UNI2 features and coordinates."""

    def __init__(self, config: dict, structure: CenteredGeneStructure) -> None:
        super().__init__()
        model = config["model"]
        n_genes = len(structure.gene_names)
        context_dim = int(model["context_dim"])
        self.gene_names = structure.gene_names
        self.conditioner = ImageSpatialConditioner(
            image_feature_dim=int(model["image_feature_dim"]),
            image_proj_dim=int(model["image_proj_dim"]),
            coord_dim=int(model["coord_dim"]),
            context_dim=context_dim,
            n_heads=int(model["n_heads"]),
            n_blocks=int(model["n_blocks"]),
            dropout=float(model["dropout"]),
            dense_threshold=int(model["dense_threshold"]),
            spatial_k=int(model["spatial_k"]),
            gex_proj_dim=int(model.get("gex_proj_dim", 256)),
            ring_embed_dim=int(model.get("ring_embed_dim", 16)),
            modality_flag_dim=int(model.get("modality_flag_dim", 16)),
        )
        hidden = int(model["decoder_hidden_dim"])
        self.decoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(hidden, n_genes),
        )
        self.within = WithinGeneRefiner(
            structure, hidden_dim=int(model["gene_structure_hidden_dim"])
        )
        self.between = BetweenSpotRefiner(
            n_genes=n_genes,
            context_dim=context_dim,
            gex_dim=int(model["refinement_gex_dim"]),
            hidden_dim=int(model["refinement_hidden_dim"]),
            k_neighbors=int(model["refinement_k"]),
        )
        self.composition_logits = nn.Parameter(torch.zeros(2))

    def encode(self, inputs: InputBatch) -> torch.Tensor:
        """Return the image-conditioned spatial context for every spot."""
        return self.conditioner(inputs)

    def predict_all(self, inputs: InputBatch) -> dict[str, torch.Tensor]:
        """Return the final prediction and intermediate refinement paths."""
        context = self.encode(inputs)
        base = self.decoder(context)
        within = self.within(base)
        between = self.between(base, context, inputs.coordinates)
        gates = torch.sigmoid(self.composition_logits)
        # Both refiners learn residual corrections to the same base prediction.
        combined = base + gates[0] * (within - base) + gates[1] * (between - base)
        query = inputs.query_mask
        return {
            "expression": combined[query],
            "context": context[query],
            "base": base[query],
            "within": within[query],
            "between": between[query],
            "gates": gates,
        }

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        """Return final expression for query spots."""
        return self.predict_all(inputs)["expression"]
