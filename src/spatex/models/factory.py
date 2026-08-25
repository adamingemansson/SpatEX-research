"""Construct the configured deterministic or WAE model."""

from __future__ import annotations

from torch import nn

from spatex.models.deterministic import SpatEX
from spatex.models.wae import SpatEXWAE
from spatex.structure import CenteredGeneStructure


def build_model(config: dict, structure: CenteredGeneStructure) -> nn.Module:
    """Build one model from a resolved configuration and gene structure."""
    deterministic = SpatEX(config, structure)
    kind = str(config["model"]["kind"])
    if kind == "deterministic":
        return deterministic
    if kind == "wae":
        return SpatEXWAE(config, deterministic)
    raise ValueError(f"unsupported model kind: {kind}")
