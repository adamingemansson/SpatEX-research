"""Evaluate SpatEX models on complete held-out slides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spatex.checkpoint import load_checkpoint
from spatex.config import load_config
from spatex.data import PreparedManifest, SlideRecord
from spatex.metrics import mean_ci, panel_metrics
from spatex.models.factory import build_model
from spatex.models.wae import SpatEXWAE
from spatex.panels import load_panels
from spatex.structure import load_structure
from spatex.structured_metrics import structured_panel_metrics
from spatex.tensorboard import example_map


def _macro(rows: dict) -> dict:
    """Aggregate each path and panel across held-out slides."""
    result: dict = {}
    metrics = (
        "gene_pcc",
        "rmse",
        "mae",
        "spot_profile_pcc",
        "flattened_pcc",
        "amplitude_ratio",
        "prediction_effective_rank",
        "target_effective_rank",
    )
    for path_name, panel_rows in rows.items():
        result[path_name] = {}
        for panel_name, values in panel_rows.items():
            summary: dict[str, float | int | list[float]] = {}
            for metric in metrics:
                mean, lower, upper = mean_ci([row[metric] for row in values])
                summary[metric] = mean
                summary[f"{metric}_95ci"] = [lower, upper]
            summary["n_slides"] = len(values)
            summary["n_genes"] = int(values[0]["n_genes"])
            result[path_name][panel_name] = summary
    return result


@torch.no_grad()
def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    split: str | None = None,
    wae_draws: int | None = None,
    panel_file: str | Path | None = None,
) -> Path:
    """Evaluate one checkpoint and save slide, panel, and per-gene results."""
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    manifest = PreparedManifest.load(config["data"]["manifest"])
    structure = load_structure(config["data"]["structure"])
    model = build_model(config, structure).to(device).eval()
    payload = load_checkpoint(checkpoint_path, model, config, manifest.gene_names)
    split = split or str(config["evaluation"]["split"])
    wae_draws = int(wae_draws or config["evaluation"]["wae_draws"])
    panel_file = panel_file or config.get("evaluation", {}).get("panel_file")
    panels = load_panels(panel_file, manifest.gene_names)
    slides = manifest.for_split(split)
    if not slides:
        raise ValueError(f"split has no slides: {split}")

    reports: list[dict[str, object]] = []
    aggregate: dict = {}
    per_gene: dict[str, list[np.ndarray]] = {}
    marker_arrays: dict[str, np.ndarray] = {}
    marker_metadata: list[dict[str, object]] = []
    marker_indices = example_map(config.get("tensorboard", {}), manifest.gene_names)
    sample_ids: list[str] = []
    organs: list[str] = []
    for item, spec in enumerate(slides, 1):
        record = SlideRecord.load(spec, len(manifest.gene_names))
        inputs, target = record.whole_slide(device)
        if isinstance(model, SpatEXWAE):
            sampled = model.sample(inputs, wae_draws)
            predictions = {
                "deterministic": sampled["deterministic"],
                "predictive_mean": sampled["predictive_mean"],
                "sample_0": sampled["samples"][0],
                **model.controls(inputs, target),
            }
            uncertainty = {
                "mean_predictive_variance": float(
                    sampled["predictive_variance"].mean().cpu()
                ),
                "mean_predictive_std": float(
                    torch.sqrt(sampled["predictive_variance"]).mean().cpu()
                ),
            }
        else:
            predictions = {"deterministic": model(inputs)}
            uncertainty = {}

        path_reports: dict[str, object] = {}
        for path_name, prediction in predictions.items():
            metrics, genes = panel_metrics(prediction, target, panels)
            structured = structured_panel_metrics(
                prediction.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                inputs.coordinates.detach().cpu().numpy(),
                structure.per_gene_scale.numpy(),
                panels,
            )
            path_reports[path_name] = {"panels": metrics, "structured": structured}
            for panel_name, values in metrics.items():
                aggregate.setdefault(path_name, {}).setdefault(panel_name, []).append(values)
            per_gene.setdefault(path_name, []).append(genes["all_genes"])
        selected = marker_indices.get(spec.sample_id, ())
        if selected:
            marker_arrays[f"coordinates__{spec.sample_id}"] = (
                inputs.coordinates.detach().cpu().numpy().astype(np.float32)
            )
            marker_arrays[f"target__{spec.sample_id}"] = (
                target[:, selected].detach().cpu().numpy().astype(np.float32)
            )
            for path_name, prediction in predictions.items():
                marker_arrays[f"prediction__{path_name}__{spec.sample_id}"] = (
                    prediction[:, selected].detach().cpu().numpy().astype(np.float32)
                )
            marker_metadata.append(
                {
                    "sample_id": spec.sample_id,
                    "organ": spec.organ,
                    "genes": [manifest.gene_names[index] for index in selected],
                    "paths": list(predictions),
                }
            )
        reports.append(
            {
                "sample_id": spec.sample_id,
                "organ": spec.organ,
                "n_spots": len(target),
                "paths": path_reports,
                **uncertainty,
            }
        )
        sample_ids.append(spec.sample_id)
        organs.append(spec.organ)
        print(f"evaluation {item}/{len(slides)}: {spec.sample_id}", flush=True)

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    per_gene_path = output.with_suffix(".per_gene.npz")
    marker_path = output.with_suffix(".markers.npz")
    marker_metadata_path = output.with_suffix(".markers.json")
    np.savez_compressed(
        per_gene_path,
        gene_names=np.asarray(manifest.gene_names, dtype=str),
        sample_ids=np.asarray(sample_ids, dtype=str),
        organs=np.asarray(organs, dtype=str),
        **{
            f"pcc__{name}": np.stack(values).astype(np.float32)
            for name, values in per_gene.items()
        },
    )
    np.savez_compressed(marker_path, **marker_arrays)
    marker_metadata_path.write_text(json.dumps(marker_metadata, indent=2) + "\n")
    output.write_text(
        json.dumps(
            {
                "kind": "spatex_evaluation",
                "version": 2,
                "model_kind": config["model"]["kind"],
                "prior": config["model"].get("prior"),
                "split": split,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_step": int(payload["step"]),
                "wae_draws": wae_draws if isinstance(model, SpatEXWAE) else None,
                "primary_metric": "macro gene-wise spatial PCC",
                "panel_file": str(Path(panel_file).expanduser().resolve()) if panel_file else None,
                "per_gene_file": str(per_gene_path),
                "marker_file": str(marker_path),
                "marker_metadata_file": str(marker_metadata_path),
                "inference_contract": {
                    "query_gex_visible": False,
                    "image_visible": True,
                    "coordinates_visible": bool(
                        config["model"].get("use_coordinates", True)
                    ),
                },
                "macro_by_slide": _macro(aggregate),
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
    parser = argparse.ArgumentParser(description="Whole-slide SpatEX evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split")
    parser.add_argument("--wae-draws", type=int)
    parser.add_argument("--panel-file")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.output,
        args.split,
        args.wae_draws,
        args.panel_file,
    )


if __name__ == "__main__":
    main()
