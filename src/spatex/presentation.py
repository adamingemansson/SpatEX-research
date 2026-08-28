"""Export presentation figures from completed SpatEX evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PANEL_ORDER = (
    "all_genes",
    "train_log1p_variance_top50",
    "train_log1p_variance_top200",
    "train_within_slide_variance_top50",
    "train_within_slide_variance_top200",
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _reports(arguments: list[str]) -> list[tuple[str, Path, dict]]:
    """Parse repeated label=path report arguments."""
    reports = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("reports must use label=/path/to/report.json")
        label, raw_path = argument.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text())
        if payload.get("kind") != "spatex_evaluation" or payload.get("version") != 2:
            raise ValueError(f"{path}: unsupported evaluation report")
        reports.append((label, path, payload))
    if not reports:
        raise ValueError("at least one report is required")
    return reports


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write rows with the union of encountered fields."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def primary_figure(reports: list[tuple[str, Path, dict]], output: Path) -> None:
    """Export the primary PCC/RMSE table and confidence-interval plot."""
    plt = _pyplot()
    rows = []
    for label, path, report in reports:
        panels = report["macro_by_slide"]["deterministic"]
        for panel in PANEL_ORDER:
            if panel not in panels:
                continue
            values = panels[panel]
            rows.append(
                {
                    "method": label,
                    "panel": panel,
                    "gene_pcc": values["gene_pcc"],
                    "gene_pcc_low": values["gene_pcc_95ci"][0],
                    "gene_pcc_high": values["gene_pcc_95ci"][1],
                    "rmse": values["rmse"],
                    "rmse_low": values["rmse_95ci"][0],
                    "rmse_high": values["rmse_95ci"][1],
                    "report": str(path),
                }
            )
    _write_rows(output / "primary_metrics.tsv", rows)
    methods = [label for label, _, _ in reports]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(rows) * 0.20)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for panel_index, panel in enumerate(PANEL_ORDER):
        for method_index, method in enumerate(methods):
            match = next(
                (row for row in rows if row["panel"] == panel and row["method"] == method),
                None,
            )
            if match is None:
                continue
            y = panel_index + (method_index - (len(methods) - 1) / 2) * 0.08
            for axis, metric in zip(axes, ("gene_pcc", "rmse"), strict=True):
                axis.errorbar(
                    match[metric],
                    y,
                    xerr=[[match[metric] - match[f"{metric}_low"]], [match[f"{metric}_high"] - match[metric]]],
                    fmt="o",
                    color=colors[method_index],
                    capsize=2,
                    label=method if panel_index == 0 else None,
                )
    labels = [value.replace("train_", "").replace("_", " ") for value in PANEL_ORDER]
    for axis, title in zip(axes, ("Gene-wise spatial PCC ↑", "RMSE ↓"), strict=True):
        axis.set_yticks(range(len(PANEL_ORDER)), labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "primary_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "primary_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def organ_figure(reports: list[tuple[str, Path, dict]], output: Path) -> None:
    """Export an organ-by-method all-gene PCC heatmap."""
    plt = _pyplot()
    rows = []
    organs = sorted({slide["organ"] for _, _, report in reports for slide in report["slides"]})
    methods = [label for label, _, _ in reports]
    matrix = np.full((len(organs), len(methods)), np.nan)
    for method_index, (label, path, report) in enumerate(reports):
        grouped: dict[str, list[float]] = defaultdict(list)
        for slide in report["slides"]:
            value = slide["paths"]["deterministic"]["panels"]["all_genes"]["gene_pcc"]
            grouped[slide["organ"]].append(value)
            rows.append(
                {
                    "method": label,
                    "sample_id": slide["sample_id"],
                    "organ": slide["organ"],
                    "gene_pcc": value,
                    "report": str(path),
                }
            )
        for organ_index, organ in enumerate(organs):
            matrix[organ_index, method_index] = np.mean(grouped[organ])
    _write_rows(output / "organ_by_slide.tsv", rows)
    fig, axis = plt.subplots(figsize=(max(8, len(methods) * 1.15), 5.5))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    axis.set_yticks(range(len(organs)), organs)
    for row in range(len(organs)):
        for column in range(len(methods)):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white" if matrix[row, column] < np.nanmedian(matrix) else "black", fontsize=8)
    axis.set_title("All-gene spatial PCC by held-out organ")
    fig.colorbar(image, ax=axis, label="macro gene-wise PCC")
    fig.tight_layout()
    fig.savefig(output / "organ_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "organ_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def structured_figure(reports: list[tuple[str, Path, dict]], output: Path) -> None:
    """Export mean structured metrics for the fixed 200-gene panels."""
    plt = _pyplot()
    metrics = {
        "coexpression": ("coexpression", "pcc"),
        "Moran": ("moran", "pcc"),
        "local gradient": ("gradient_local", "pcc"),
        "wide gradient": ("gradient_wide", "pcc"),
        "SSIM": ("spatial_ssim", "mean"),
    }
    panel = "train_within_slide_variance_top200"
    rows = []
    for label, path, report in reports:
        values: dict[str, list[float]] = defaultdict(list)
        for slide in report["slides"]:
            structured = slide["paths"]["deterministic"].get("structured", {}).get(panel, {})
            for name, (section, field) in metrics.items():
                if section in structured and np.isfinite(structured[section].get(field, np.nan)):
                    values[name].append(structured[section][field])
        for name in metrics:
            if values[name]:
                rows.append(
                    {
                        "method": label,
                        "panel": panel,
                        "metric": name,
                        "mean": float(np.mean(values[name])),
                        "n_slides": len(values[name]),
                        "report": str(path),
                    }
                )
    _write_rows(output / "structured_metrics.tsv", rows)
    if not rows:
        return
    methods = [label for label, _, _ in reports]
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for index, method in enumerate(methods):
        heights = [next((row["mean"] for row in rows if row["method"] == method and row["metric"] == metric), np.nan) for metric in metrics]
        axis.bar(x + (index - (len(methods) - 1) / 2) * width, heights, width, label=method)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, metrics)
    axis.set_ylabel("agreement")
    axis.set_title("Preservation of within- and between-spot structure")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "structured_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "structured_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def marker_figures(label: str, path: Path, report: dict, output: Path) -> None:
    """Export target, prediction, and error maps for preregistered markers."""
    plt = _pyplot()
    metadata_path = Path(report["marker_metadata_file"])
    arrays_path = Path(report["marker_file"])
    metadata = json.loads(metadata_path.read_text())
    arrays = np.load(arrays_path, allow_pickle=False)
    marker_root = output / "markers" / label
    marker_root.mkdir(parents=True, exist_ok=True)
    for slide in metadata:
        sample = slide["sample_id"]
        coordinates = arrays[f"coordinates__{sample}"]
        target = arrays[f"target__{sample}"]
        prediction = arrays[f"prediction__deterministic__{sample}"]
        # Slight overlap makes the tissue field readable without obscuring boundaries.
        marker_size = max(7.0, min(22.0, 26000.0 / len(coordinates)))
        for gene_index, gene in enumerate(slide["genes"]):
            truth = target[:, gene_index]
            pred = prediction[:, gene_index]
            upper = float(max(np.quantile(truth, 0.995), np.quantile(pred, 0.995), 1e-6))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, values, title, cmap, vmax in (
                (axes[0], truth, "Target", "viridis", upper),
                (axes[1], pred, "Prediction", "viridis", upper),
                (axes[2], np.abs(pred - truth), "Absolute error", "magma", None),
            ):
                plot = axis.scatter(coordinates[:, 0], -coordinates[:, 1], c=values, s=marker_size, cmap=cmap, vmin=0, vmax=vmax, linewidths=0)
                axis.set_title(title)
                axis.set_aspect("equal")
                axis.axis("off")
                fig.colorbar(plot, ax=axis, fraction=0.035, pad=0.02)
            centered_truth = truth - truth.mean()
            centered_pred = pred - pred.mean()
            denominator = np.sqrt(np.sum(centered_truth**2) * np.sum(centered_pred**2))
            pcc = float(np.sum(centered_truth * centered_pred) / denominator) if denominator > 1e-12 else float("nan")
            rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
            fig.suptitle(f"{label} | {slide['organ']} {sample} | {gene}\nPCC={pcc:.3f}  RMSE={rmse:.3f}")
            fig.tight_layout()
            fig.savefig(marker_root / f"{slide['organ']}__{sample}__{gene}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def export(reports: list[tuple[str, Path, dict]], output: Path) -> Path:
    """Build the complete presentation figure directory."""
    output.mkdir(parents=True, exist_ok=False)
    primary_figure(reports, output)
    organ_figure(reports, output)
    structured_figure(reports, output)
    for label, path, report in reports:
        marker_figures(label, path, report, output)
    (output / "sources.json").write_text(
        json.dumps({label: str(path) for label, path, _ in reports}, indent=2) + "\n"
    )
    return output


def main() -> None:
    """Export figures from one or more named reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = export(_reports(args.report), Path(args.output_dir).expanduser().resolve())
    print(output)


if __name__ == "__main__":
    main()
