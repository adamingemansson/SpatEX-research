"""Evaluate models on complete held-out slides.

Deterministic predictions, deployable WAE samples, and posterior diagnostics
are reported as separate paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spatex.checkpoint import load_checkpoint
from spatex.config import load_config
from spatex.data import PreparedManifest, SlideRecord
from spatex.losses import pearson_correlation, rmse
from spatex.models.factory import build_model
from spatex.models.wae import SpatEXWAE
from spatex.structure import load_structure


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Calculate flattened PCC, RMSE, and MAE for one slide."""
    return {
        "pcc": float(pearson_correlation(prediction, target).cpu()),
        "rmse": float(rmse(prediction, target).cpu()),
        "mae": float(torch.mean(torch.abs(prediction - target)).cpu()),
    }


@torch.no_grad()
def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    split: str | None = None,
    wae_draws: int | None = None,
) -> Path:
    """Evaluate every slide in a split and write a self-describing JSON report."""
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    manifest = PreparedManifest.load(config["data"]["manifest"])
    structure = load_structure(config["data"]["structure"])
    model = build_model(config, structure).to(device).eval()
    payload = load_checkpoint(checkpoint_path, model, config, manifest.gene_names)
    split = split or str(config["evaluation"]["split"])
    wae_draws = int(wae_draws or config["evaluation"]["wae_draws"])
    slides = manifest.for_split(split)
    reports = []
    aggregate: dict[str, list[dict[str, float]]] = {}
    for item, spec in enumerate(slides, 1):
        record = SlideRecord.load(spec, len(manifest.gene_names))
        inputs, target = record.whole_slide(device)
        predictions: dict[str, torch.Tensor]
        variances: dict[str, float] = {}
        if isinstance(model, SpatEXWAE):
            # Keep deployable prior draws separate from GEX-dependent diagnostics.
            sampled = model.sample(inputs, wae_draws)
            predictions = {
                "deterministic": sampled["deterministic"],
                "predictive_mean": sampled["predictive_mean"],
                "sample_0": sampled["samples"][0],
                **model.controls(inputs, target),
            }
            variances = {
                "mean_predictive_variance": float(
                    sampled["predictive_variance"].mean().cpu()
                ),
                "mean_predictive_std": float(
                    torch.sqrt(sampled["predictive_variance"]).mean().cpu()
                ),
            }
        else:
            predictions = {"deterministic": model(inputs)}
        path_metrics = {name: _metrics(value, target) for name, value in predictions.items()}
        for name, values in path_metrics.items():
            aggregate.setdefault(name, []).append(values)
        reports.append(
            {
                "sample_id": spec.sample_id,
                "organ": spec.organ,
                "n_spots": len(target),
                "paths": path_metrics,
                **variances,
            }
        )
        print(f"evaluation {item}/{len(slides)}: {spec.sample_id}", flush=True)
    macro = {
        name: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("pcc", "rmse", "mae")
        }
        for name, rows in aggregate.items()
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "kind": "spatex_evaluation",
                "version": 1,
                "model_kind": config["model"]["kind"],
                "prior": config["model"].get("prior"),
                "split": split,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_step": int(payload["step"]),
                "wae_draws": wae_draws if isinstance(model, SpatEXWAE) else None,
                "inference_contract": {
                    "query_gex_visible": False,
                    "image_visible": True,
                    "coordinates_visible": True,
                },
                "macro_by_slide": macro,
                "slides": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"evaluation saved to {output}")
    return output


def main() -> None:
    """Run whole-slide evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Whole-slide model evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split")
    parser.add_argument("--wae-draws", type=int)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.split, args.wae_draws)


if __name__ == "__main__":
    main()
