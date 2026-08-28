"""Define the deterministic SpatEX predictor.

A full-GEX decoder is followed by parallel within-gene and between-spot
residual refiners whose contributions are learned through two gates.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spatex.inputs import InputBatch
from spatex.models.conditioner import ImageSpatialConditioner
from spatex.models.legacy import (
    LegacyBetweenSpotRefiner,
    LegacyImageSpatialConditioner,
)
from spatex.models.refinement import BetweenSpotRefiner, WithinGeneRefiner
from spatex.structure import CenteredGeneStructure


class SpatEX(nn.Module):
    """Predict full spatial expression from UNI2 features and coordinates."""

    def __init__(self, config: dict, structure: CenteredGeneStructure) -> None:
        super().__init__()
        model = config["model"]
        n_genes = len(structure.gene_names)
        context_dim = int(model["context_dim"])
        self.use_within_refiner = bool(model.get("use_within_refiner", True))
        self.use_between_refiner = bool(model.get("use_between_refiner", True))
        self.gene_names = structure.gene_names
        self.legacy_parallel_gated = bool(model.get("legacy_parallel_gated", False))
        conditioner_arguments = dict(
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
        if self.legacy_parallel_gated:
            self.conditioner = LegacyImageSpatialConditioner(**conditioner_arguments)
        else:
            self.conditioner = ImageSpatialConditioner(
                **conditioner_arguments,
                use_coordinates=bool(model.get("use_coordinates", True)),
                use_spatial_attention=bool(model.get("use_spatial_attention", True)),
            )
        hidden = int(model["decoder_hidden_dim"])
        if self.legacy_parallel_gated:
            self.decoder = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_genes),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(context_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(model["dropout"])),
                nn.Linear(hidden, n_genes),
            )
        self.within = WithinGeneRefiner(
            structure, hidden_dim=int(model["gene_structure_hidden_dim"])
        )
        between_type = (
            LegacyBetweenSpotRefiner
            if self.legacy_parallel_gated
            else BetweenSpotRefiner
        )
        self.between = between_type(
            n_genes=n_genes,
            context_dim=context_dim,
            gex_dim=int(model["refinement_gex_dim"]),
            hidden_dim=int(model["refinement_hidden_dim"]),
            k_neighbors=int(model["refinement_k"]),
        )
        self.refinement_steps = int(model.get("refinement_steps", 1))
        if self.refinement_steps < 1:
            raise ValueError("refinement_steps must be positive")
        gate_weight = float(model.get("gate_initial_weight", 0.5))
        if not 0.0 < gate_weight < 1.0:
            raise ValueError("gate_initial_weight must be between zero and one")
        gate_logit = math.log(gate_weight / (1.0 - gate_weight))
        self.within_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        self.between_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        if not self.use_within_refiner:
            self.within.requires_grad_(False)
            self.within_gate_logit.requires_grad_(False)
        if not self.use_between_refiner:
            self.between.requires_grad_(False)
            self.between_gate_logit.requires_grad_(False)

    def encode(self, inputs: InputBatch) -> torch.Tensor:
        """Return the image-conditioned spatial context for every spot."""
        return self.conditioner(inputs)

    def predict_all(self, inputs: InputBatch) -> dict[str, torch.Tensor]:
        """Return the final prediction and intermediate refinement paths."""
        context = self.encode(inputs)
        base = self.decoder(context)
        within = self.within(base) if self.use_within_refiner else base
        between = base
        if self.use_between_refiner:
            for _ in range(self.refinement_steps):
                between = self.between(between, context, inputs.coordinates)
        within_gate = (
            torch.sigmoid(self.within_gate_logit)
            if self.use_within_refiner
            else base.new_zeros(())
        )
        between_gate = (
            torch.sigmoid(self.between_gate_logit)
            if self.use_between_refiner
            else base.new_zeros(())
        )
        gates = torch.stack((within_gate, between_gate))
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
