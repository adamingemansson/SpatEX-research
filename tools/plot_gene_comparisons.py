#!/usr/bin/env python3
"""Plot fixed target-versus-prediction maps across all evaluated models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def pyplot():
    """Load a non-interactive Matplotlib backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_source(argument: str) -> dict:
    """Load one label=report marker bundle."""
    label, raw_path = argument.split("=", 1)
    report_path = Path(raw_path).expanduser().resolve()
    report = json.loads(report_path.read_text())
    marker_path = Path(report["marker_file"])
    metadata_path = Path(report["marker_metadata_file"])
    if not marker_path.is_file():
        marker_path = report_path.parent / marker_path.name
    if not metadata_path.is_file():
        metadata_path = report_path.parent / metadata_path.name
    metadata = json.loads(metadata_path.read_text())
    primary = report.get("primary_path")
    if primary is None:
        primary = "pretrained_zero_shot" if "stpath" in label.lower() else "deterministic"
    return {
        "label": label,
        "report": report,
        "arrays": np.load(marker_path, allow_pickle=False),
        "metadata": {row["sample_id"]: row for row in metadata},
        "primary": primary,
    }


def comparison_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate metrics shown with each spatial prediction map."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[finite], target[finite]
    if len(target) < 2:
        return {"pcc": float("nan"), "spearman": float("nan"), "rmse": float("nan")}
    pcc = 0.0 if np.std(prediction) < 1e-12 else float(np.corrcoef(prediction, target)[0, 1])
    pred_rank, target_rank = rankdata(prediction), rankdata(target)
    spearman = 0.0 if np.std(pred_rank) < 1e-12 else float(np.corrcoef(pred_rank, target_rank)[0, 1])
    return {
        "pcc": pcc,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "n_spots": int(len(target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="label=/path/report.json")
    parser.add_argument("--gene-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models-per-page", type=int, default=5)
    parser.add_argument(
        "--sample-policy", choices=("all", "top_target_variance"), default="all",
    )
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()
    if args.models_per_page < 1:
        raise ValueError("models-per-page must be positive")

    sources = [load_source(value) for value in args.source]
    selection = json.loads(Path(args.gene_selection).read_text())
    ranked_genes = [
        (group, row["gene"], int(row["rank"]))
        for group, group_rows in selection["selection"].items()
        for row in group_rows
    ]
    sample_ids = sorted(set.intersection(*(set(source["metadata"]) for source in sources)))
    if not sample_ids:
        raise ValueError("marker bundles have no common slides")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    metric_rows = []
    plt = pyplot()
    plot_items = []
    reference = sources[0]
    for group, gene, rank in ranked_genes:
        candidates = []
        for sample_id in sample_ids:
            meta = reference["metadata"][sample_id]
            if gene not in meta["genes"]:
                continue
            index = meta["genes"].index(gene)
            target = reference["arrays"][f"target__{sample_id}"][:, index]
            candidates.append((float(np.var(target)), sample_id))
        if args.sample_policy == "top_target_variance" and candidates:
            candidates = [max(candidates)]
        plot_items.extend((group, gene, rank, sample_id) for _, sample_id in candidates)

    for group, gene, rank, sample_id in plot_items:
        reference = sources[0]
        reference_meta = reference["metadata"][sample_id]
        coords = reference["arrays"][f"coordinates__{sample_id}"]
        target_matrix = reference["arrays"][f"target__{sample_id}"]
        target_lookup = {gene: i for i, gene in enumerate(reference_meta["genes"])}
        organ = reference_meta.get("organ", "unknown")
        point_size = max(2.0, min(9.0, 18000.0 / max(len(coords), 1)))
        target = target_matrix[:, target_lookup[gene]]
        predictions = []
        for source in sources:
            meta = source["metadata"][sample_id]
            if gene not in meta["genes"]:
                continue
            index = meta["genes"].index(gene)
            key = f"prediction__{source['primary']}__{sample_id}"
            source_coords = source["arrays"][f"coordinates__{sample_id}"]
            values = source["arrays"][key][:, index]
            source_target = source["arrays"][f"target__{sample_id}"][:, index]
            metrics = comparison_metrics(values, source_target)
            predictions.append(
                (source["label"], values, source_coords, metrics)
            )
            metric_rows.append({
                "category": group,
                "rank": rank,
                "gene": gene,
                "sample_id": sample_id,
                "organ": organ,
                "model": source["label"],
                **metrics,
            })
        all_values = np.concatenate([target, *(values for _, values, _, _ in predictions)])
        vmax = max(float(np.quantile(all_values[np.isfinite(all_values)], 0.995)), 1e-6)
        pages = math.ceil(len(predictions) / args.models_per_page)
        for page in range(pages):
            subset = predictions[
                page * args.models_per_page : (page + 1) * args.models_per_page
            ]
            figure, axes = plt.subplots(
                1, len(subset) + 1, figsize=(3.4 * (len(subset) + 1), 3.8),
                constrained_layout=True,
            )
            panels = [("Target", target, coords, None), *subset]
            for axis, (label, values, panel_coords, metrics) in zip(
                np.atleast_1d(axes), panels, strict=True
            ):
                image = axis.scatter(
                    panel_coords[:, 0], -panel_coords[:, 1], c=values, s=point_size,
                    cmap="viridis", vmin=0.0, vmax=vmax, linewidths=0,
                )
                title = label if metrics is None else f"{label}\nPCC={metrics['pcc']:.3f}"
                axis.set_title(title, fontsize=9)
                axis.set_aspect("equal")
                axis.axis("off")
            figure.colorbar(image, ax=np.atleast_1d(axes).tolist(), fraction=0.018, pad=0.01)
            figure.suptitle(
                f"{group} rank {rank} | {gene} | {organ} {sample_id} | page {page + 1}/{pages}",
                fontsize=11,
            )
            folder = output / group
            folder.mkdir(exist_ok=True)
            path = folder / f"rank{rank:02d}__{organ}__{sample_id}__{gene}__page{page + 1}.png"
            figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(figure)
            rows.append({
                "category": group,
                "rank": rank,
                "gene": gene,
                "sample_id": sample_id,
                "organ": organ,
                "page": page + 1,
                "models": "; ".join(label for label, _, _, _ in subset),
                "target_variance": float(np.var(target)),
                "path": str(path),
            })
    with (output / "plot_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "plot_panel_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps({"plots": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
