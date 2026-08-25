"""Run whole-slide inference from prepared UNI2 inputs.

The input loader accepts image features, coordinates, availability, and
barcodes, but never measured expression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spatex.checkpoint import load_checkpoint
from spatex.config import load_config
from spatex.inputs import InputBatch
from spatex.models.factory import build_model
from spatex.models.wae import SpatEXWAE
from spatex.structure import load_structure


def load_inference_slide(path: str | Path, device: torch.device) -> tuple[InputBatch, np.ndarray]:
    """Load one slide without accepting measured expression as input."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        required = {"image_features", "coordinates", "image_available", "barcodes"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing inference fields {sorted(missing)}")
        features = payload["image_features"].astype(np.float32)
        coordinates = payload["coordinates"].astype(np.float32)
        available = payload["image_available"].astype(bool)
        barcodes = payload["barcodes"].astype(str)
    n = len(features)
    if coordinates.shape != (n, 2) or available.shape != (n,) or barcodes.shape != (n,):
        raise ValueError(f"{path}: inference fields do not share the same spot dimension")
    inputs = InputBatch(
        sample_id=path.stem,
        image_features=torch.from_numpy(features).to(device),
        coordinates=torch.from_numpy(coordinates).to(device),
        image_available=torch.from_numpy(available).to(device),
        query_mask=torch.ones(n, dtype=torch.bool, device=device),
    )
    return inputs, barcodes


@torch.no_grad()
def predict(
    config_path: str | Path,
    checkpoint_path: str | Path,
    slide_path: str | Path,
    output_path: str | Path,
    draws: int | None = None,
) -> Path:
    """Load a model and save deterministic or WAE predictions to NPZ."""
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    structure = load_structure(config["data"]["structure"])
    model = build_model(config, structure).to(device).eval()
    load_checkpoint(checkpoint_path, model, config, structure.gene_names)
    inputs, barcodes = load_inference_slide(slide_path, device)
    if isinstance(model, SpatEXWAE):
        # Save individual draws so uncertainty can be inspected after inference.
        result = model.sample(inputs, int(draws or config["evaluation"]["wae_draws"]))
        arrays = {
            "deterministic": result["deterministic"].cpu().numpy(),
            "predictive_mean": result["predictive_mean"].cpu().numpy(),
            "predictive_variance": result["predictive_variance"].cpu().numpy(),
            "samples": result["samples"].cpu().numpy(),
        }
    else:
        arrays = {"deterministic": model(inputs).cpu().numpy()}
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        barcodes=barcodes,
        gene_names=np.asarray(structure.gene_names),
        query_gex_visible=np.asarray(False),
    )
    print(f"predictions saved to {output_path}")
    return output_path


def main() -> None:
    """Run whole-slide inference from the command line."""
    parser = argparse.ArgumentParser(description="GEX-free whole-slide inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int)
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.slide, args.output, args.draws)


if __name__ == "__main__":
    main()
