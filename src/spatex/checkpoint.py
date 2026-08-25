"""Save and restore SpatEX training state.

Configuration and gene-order hashes prevent incompatible checkpoints from
being loaded silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from spatex.structure import gene_hash


def config_hash(config: dict) -> str:
    """Return a stable hash of the resolved training configuration."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    gene_names: tuple[str, ...],
    step: int,
    validation: dict[str, float],
) -> None:
    """Save model, optimizer, step, and validation state in one checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config_hash": config_hash(config),
            "gene_hash": gene_hash(gene_names),
            "step": int(step),
            "validation": validation,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict,
    gene_names: tuple[str, ...],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load a compatible checkpoint and optionally restore its optimizer."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    # Refuse silent reuse with a different architecture or gene ordering.
    if payload.get("config_hash") != config_hash(config):
        raise ValueError("checkpoint configuration does not match")
    if payload.get("gene_hash") != gene_hash(gene_names):
        raise ValueError("checkpoint gene order does not match")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
