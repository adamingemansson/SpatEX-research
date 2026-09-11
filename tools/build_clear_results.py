#!/usr/bin/env python3
"""Create readable metric tables and comparison figures from metrics_long.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PANEL_NAMES = {
    "all_genes": "all_genes",
    "train_log1p_variance_top50": "hvg_50",
    "train_log1p_variance_top200": "hvg_200",
    "train_within_slide_variance_top50": "within_50",
    "train_within_slide_variance_top200": "within_200",
    "HVG-50": "hvg_50",
    "HVG-200": "hvg_200",
    "within-50": "within_50",
    "within-200": "within_200",
}

METRICS = {
    "spatial_gene_pcc": ("pcc", "gene_pcc"),
    "spatial_gene_spearman": ("spearman",),
    "rmse": ("rmse",),
    "mae": ("mae",),
    "r2": ("r2",),
    "expression_auc": ("auc",),
    "median_gene_pcc": ("median_gene_pcc",),
    "fraction_gene_pcc_gt_0_1": ("fraction_gene_pcc_gt_0_1",),
    "spot_profile_pcc": (
        "structured.spot_profile.mean_spot_profile_pcc",
        "structured.spot_profile.pcc",
        "spot_profile_pcc",
    ),
    "coexpression_pcc": (
        "structured.coexpression.pcc",
        "structured.coexpression.correlation_matrix_pcc",
    ),
    "coexpression_mae": (
        "structured.coexpression.mae",
        "structured.coexpression.correlation_matrix_mae",
    ),
    "moran_pcc": ("structured.moran.pcc", "structured.moran_local.moran_i_pcc"),
    "moran_mae": ("structured.moran.mae", "structured.moran_local.moran_i_mae"),
    "moran_target_mean": (
        "structured.moran.target_mean",
        "structured.moran_local.target_mean_moran_i",
    ),
    "moran_prediction_mean": (
        "structured.moran.prediction_mean",
        "structured.moran_local.predicted_mean_moran_i",
    ),
    "local_gradient_pcc": (
        "structured.gradient_local.pcc",
        "structured.gradient_local.signed_gradient_pcc",
    ),
    "local_gradient_gene_pcc": (
        "structured.gradient_local.mean_gene_pcc",
        "structured.gradient_local.mean_per_gene_gradient_pcc",
    ),
    "local_gradient_energy_ratio": (
        "structured.gradient_local.energy_ratio",
        "structured.gradient_local.gradient_energy_ratio",
    ),
    "local_gradient_sign_agreement": (
        "structured.gradient_local.sign_agreement",
        "structured.gradient_local.sign_agreement_nontrivial",
    ),
    "wide_gradient_pcc": (
        "structured.gradient_wide.pcc",
        "structured.gradient_wide.signed_gradient_pcc",
    ),
    "wide_gradient_gene_pcc": (
        "structured.gradient_wide.mean_gene_pcc",
        "structured.gradient_wide.mean_per_gene_gradient_pcc",
    ),
    "wide_gradient_energy_ratio": (
        "structured.gradient_wide.energy_ratio",
        "structured.gradient_wide.gradient_energy_ratio",
    ),
    "wide_gradient_sign_agreement": (
        "structured.gradient_wide.sign_agreement",
        "structured.gradient_wide.sign_agreement_nontrivial",
    ),
    "spatial_ssim": (
        "structured.spatial_ssim.mean",
        "structured.spatial_ssim.mean_per_gene_ssim",
    ),
}

DIAGNOSTICS = {
    "amplitude_std_ratio": "diagnostic.amplitude.median_std_ratio",
    "effective_rank_ratio": "diagnostic.template_reuse.entropy_effective_rank_ratio",
    "prediction_effective_rank": "diagnostic.template_reuse.predicted_entropy_effective_rank",
    "target_effective_rank": "diagnostic.template_reuse.target_entropy_effective_rank",
    "exact_spatial_pcc": "diagnostic.spatial_resolution.exact_pcc",
    "blur1_spatial_pcc": "diagnostic.spatial_resolution.blur1_pcc",
    "blur2_spatial_pcc": "diagnostic.spatial_resolution.blur2_pcc",
    "blur2_gain": "diagnostic.spatial_resolution.blur2_gain",
    "template_top1": "diagnostic.template_reuse.same_gene_top1_fraction",
    "template_top5": "diagnostic.template_reuse.same_gene_top5_fraction",
    "template_median_rank": "diagnostic.template_reuse.median_same_gene_rank",
    "template_max_reuse": "diagnostic.template_reuse.maximum_target_reuse_count",
    "template_unique_fraction": "diagnostic.template_reuse.unique_top_target_fraction",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_value(values: dict[str, float], aliases: tuple[str, ...]) -> float | None:
    return next((values[name] for name in aliases if name in values), None)


def primary_roles(inventory: list[dict[str, str]]) -> dict[str, str]:
    return {row["model"]: row["primary_path"] for row in inventory}


def macro_values(rows: list[dict[str, str]], roles: dict[str, str]) -> dict[tuple[str, str], dict[str, float]]:
    output: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if not row["scope"].startswith("validation_macro"):
            continue
        if row["prediction_role"] != roles.get(row["model"]):
            continue
        panel = PANEL_NAMES.get(row["panel"])
        value = numeric(row["value"])
        if panel and value is not None:
            output[(row["model"], panel)][row["metric"]] = value
    return output


def diagnostic_values(rows: list[dict[str, str]], roles: dict[str, str]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["scope"] != "validation_slide" or row["panel"] != "diagnostic_only":
            continue
        if row["prediction_role"] != roles.get(row["model"]):
            continue
        value = numeric(row["value"])
        if value is not None:
            grouped[(row["model"], row["metric"])].append(value)
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, metric), values in grouped.items():
        output[model][metric] = float(np.mean(values))
    return output


def model_table(
    inventory: list[dict[str, str]],
    macro: dict[tuple[str, str], dict[str, float]],
    diagnostics: dict[str, dict[str, float]],
) -> tuple[list[dict], list[str]]:
    fields = ["model", "model_family", "seed", "checkpoint_step", "prediction_role"]
    for panel in ("all_genes", "hvg_200", "within_200"):
        fields.extend(f"{panel}_{metric}" for metric in METRICS)
    fields.extend(DIAGNOSTICS)
    rows = []
    for item in inventory:
        model = item["model"]
        row = {field: item.get(field, "") for field in fields[:5]}
        row["prediction_role"] = item.get("primary_path", "")
        for panel in ("all_genes", "hvg_200", "within_200"):
            values = macro.get((model, panel), {})
            for metric, aliases in METRICS.items():
                row[f"{panel}_{metric}"] = selected_value(values, aliases)
        values = diagnostics.get(model, {})
        for metric, source in DIAGNOSTICS.items():
            row[metric] = values.get(source)
        rows.append(row)
    rows.sort(key=lambda row: -(row.get("all_genes_spatial_gene_pcc") or -999))
    return rows, fields


def slide_table(rows: list[dict[str, str]], roles: dict[str, str]) -> tuple[list[dict], list[str]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    families: dict[str, str] = {}
    for row in rows:
        if row["scope"] != "validation_slide" or row["panel"] == "diagnostic_only":
            continue
        if row["prediction_role"] != roles.get(row["model"]):
            continue
        panel = PANEL_NAMES.get(row["panel"])
        value = numeric(row["value"])
        if panel and value is not None:
            key = row["model"], row["sample_id"], row["organ"], panel
            grouped[key][row["metric"]] = value
            families[row["model"]] = row["model_family"]
    fields = ["model", "model_family", "sample_id", "organ", "panel", *METRICS]
    output = []
    for (model, sample, organ, panel), values in grouped.items():
        item = {
            "model": model,
            "model_family": families[model],
            "sample_id": sample,
            "organ": organ,
            "panel": panel,
        }
        for metric, aliases in METRICS.items():
            item[metric] = selected_value(values, aliases)
        output.append(item)
    output.sort(key=lambda row: (row["panel"], row["sample_id"], row["model"]))
    return output, fields


def pcc_matrix(slides: list[dict], methods: list[str]) -> tuple[list[dict], list[str]]:
    selected = [row for row in slides if row["panel"] == "all_genes"]
    samples = sorted({(row["sample_id"], row["organ"]) for row in selected})
    lookup = {(row["sample_id"], row["model"]): row["spatial_gene_pcc"] for row in selected}
    rows = []
    for sample, organ in samples:
        rows.append({
            "sample_id": sample,
            "organ": organ,
            **{method: lookup.get((sample, method)) for method in methods},
        })
    return rows, ["sample_id", "organ", *methods]


def core_model_table(models: list[dict]) -> tuple[list[dict], list[str]]:
    """Select the presentation-scale subset from the exhaustive model table."""
    fields = [
        "model", "model_family", "seed", "checkpoint_step",
        "all_genes_spatial_gene_pcc", "all_genes_spatial_gene_spearman",
        "all_genes_rmse", "all_genes_mae", "all_genes_expression_auc",
        "all_genes_spot_profile_pcc", "amplitude_std_ratio", "effective_rank_ratio",
        "hvg_200_spatial_gene_pcc", "hvg_200_spatial_gene_spearman",
        "hvg_200_moran_pcc", "hvg_200_coexpression_pcc",
        "hvg_200_local_gradient_pcc", "hvg_200_spatial_ssim",
        "within_200_spatial_gene_pcc", "within_200_spatial_gene_spearman",
        "within_200_moran_pcc", "within_200_coexpression_pcc",
        "within_200_local_gradient_pcc", "within_200_spatial_ssim",
    ]
    return [{field: row.get(field) for field in fields} for row in models], fields


def spatial_model_table(models: list[dict]) -> tuple[list[dict], list[str]]:
    """Put spatial-structure metrics in one readable row per model and panel."""
    metrics = [
        "spatial_gene_pcc", "spatial_gene_spearman", "spot_profile_pcc",
        "moran_pcc", "moran_mae", "moran_target_mean", "moran_prediction_mean",
        "coexpression_pcc", "coexpression_mae", "local_gradient_pcc",
        "local_gradient_gene_pcc", "local_gradient_energy_ratio",
        "local_gradient_sign_agreement", "wide_gradient_pcc",
        "wide_gradient_gene_pcc", "wide_gradient_energy_ratio",
        "wide_gradient_sign_agreement", "spatial_ssim",
    ]
    fields = ["model", "model_family", "checkpoint_step", "panel", *metrics]
    output = []
    for model in models:
        for panel in ("all_genes", "hvg_200", "within_200"):
            output.append({
                "model": model["model"],
                "model_family": model["model_family"],
                "checkpoint_step": model["checkpoint_step"],
                "panel": panel,
                **{metric: model.get(f"{panel}_{metric}") for metric in metrics},
            })
    return output, fields


def metric_dictionary(path: Path) -> None:
    rows = [
        ("spatial_gene_pcc", "Macro Pearson correlation of each gene across spots; primary metric.", "higher"),
        ("spatial_gene_spearman", "Macro rank correlation of each gene across spots.", "higher"),
        ("spot_profile_pcc", "Mean correlation of predicted and target gene profiles within each spot.", "higher"),
        ("rmse / mae", "Expression error in library-normalized log1p space.", "lower"),
        ("r2", "Coefficient of determination; values below zero are worse than the target mean.", "higher"),
        ("expression_auc", "ROC AUC for detecting nonzero target expression.", "higher"),
        ("moran_pcc", "Across-gene agreement between predicted and target Moran's I.", "higher"),
        ("moran_mae", "Absolute error in per-gene Moran's I.", "lower"),
        ("coexpression_pcc", "Agreement between predicted and target gene-gene correlation matrices.", "higher"),
        ("local/wide_gradient_pcc", "Agreement of signed expression differences across spatial graph edges.", "higher"),
        ("gradient_energy_ratio", "Predicted-to-target spatial gradient energy; 1 is ideal.", "closer to 1"),
        ("gradient_sign_agreement", "Fraction of nontrivial spatial gradients with the correct direction.", "higher"),
        ("spatial_ssim", "Structural similarity of spatial expression maps.", "higher"),
        ("amplitude_std_ratio", "Predicted-to-target spatial standard deviation; 1 is ideal.", "closer to 1"),
        ("effective_rank_ratio", "Predicted-to-target effective rank; 1 means matched pattern diversity.", "closer to 1"),
        ("exact/blur spatial PCC", "Spatial gene PCC before and after one/two graph smoothing passes.", "higher"),
        ("template_top1/top5", "Fraction whose predicted map retrieves the correct target gene map.", "higher"),
    ]
    write_csv(path, [dict(metric=a, definition=b, preferred_direction=c) for a, b, c in rows], ["metric", "definition", "preferred_direction"])


def heatmap(matrix_rows: list[dict], methods: list[str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray([[row.get(method) if row.get(method) is not None else np.nan for method in methods] for row in matrix_rows])
    labels = [f"{row['organ']} {row['sample_id']}" for row in matrix_rows]
    fig, axis = plt.subplots(figsize=(max(10, 0.72 * len(methods)), max(6, 0.42 * len(labels))))
    image = axis.imshow(values, cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values), aspect="auto")
    axis.set_xticks(range(len(methods)), methods, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=6,
                          color="white" if values[row, column] < np.nanmedian(values) else "black")
    axis.set_title("All-gene spatial PCC by held-out slide and method")
    fig.colorbar(image, ax=axis, label="spatial gene PCC")
    fig.tight_layout()
    fig.savefig(output / "slide_pcc_heatmap.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "slide_pcc_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-long", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.metrics_long))
    inventory = read_csv(Path(args.inventory))
    roles = primary_roles(inventory)
    macro = macro_values(rows, roles)
    diagnostics = diagnostic_values(rows, roles)
    models, model_fields = model_table(inventory, macro, diagnostics)
    core_models, core_fields = core_model_table(models)
    spatial_models, spatial_fields = spatial_model_table(models)
    slides, slide_fields = slide_table(rows, roles)
    methods = [row["model"] for row in models]
    matrix, matrix_fields = pcc_matrix(slides, methods)

    write_csv(output / "model_metrics_clear.csv", models, model_fields)
    write_csv(output / "model_metrics_core.csv", core_models, core_fields)
    write_csv(output / "model_spatial_metrics.csv", spatial_models, spatial_fields)
    write_csv(output / "slide_metrics_clear.csv", slides, slide_fields)
    write_csv(output / "slide_pcc_matrix.csv", matrix, matrix_fields)
    metric_dictionary(output / "metric_dictionary_clear.csv")
    heatmap(matrix, methods, output)
    print(json.dumps({
        "models": len(models),
        "slide_panel_rows": len(slides),
        "outputs": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
