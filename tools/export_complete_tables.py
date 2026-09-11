#!/usr/bin/env python3
"""Convert SpatEX, legacy, and STPath reports into presentation-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_FIELDS = (
    "model", "model_family", "seed", "checkpoint_step", "prediction_role",
    "scope", "sample_id", "organ", "panel", "metric", "value", "ci_low",
    "ci_high", "n", "higher_is_better", "comparison_group", "expression_space",
    "source_report", "notes",
)


def number(value: Any) -> float | None:
    """Return a finite numeric value or null."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def flatten(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested numeric dictionaries with dotted metric names."""
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(item, name))
    else:
        value = number(value)
        if value is not None:
            output[prefix] = value
    return output


def direction(metric: str) -> bool | None:
    """Return whether larger values are better for a metric."""
    lower_tokens = (
        "rmse", "mse", "mae", "error", "js_divergence", "fid", "mmd",
        "reuse_count", "same_gene_rank",
    )
    if any(token in metric for token in lower_tokens):
        return False
    higher_tokens = (
        "pcc", "spearman", "auc", "r2", "ssim", "nmi", "sign_agreement",
        "top1", "top5", "unique_top", "effective_rank", "coverage",
    )
    if any(token in metric for token in higher_tokens):
        return True
    return None


def metadata(label: str, report: dict[str, Any]) -> dict[str, Any]:
    """Resolve stable model metadata from heterogeneous reports."""
    lower = label.lower()
    match = re.search(r"seed(\d+)", lower)
    seed = int(match.group(1)) if match else None
    family = "STPath" if "stpath" in lower else "SpatEX"
    if "parallel_gated" in lower or "legacy" in lower:
        family = "legacy parallel-gated"
    if "wae" in lower:
        family = "SpatEX WAE"
    return {
        "model": label,
        "model_family": family,
        "seed": seed,
        "checkpoint_step": report.get("checkpoint_step"),
        "expression_space": report.get("expression_space", "library_normalized_log1p"),
        "source_report": str(report.get("_path", "")),
    }


def append_row(
    rows: list[dict[str, Any]], base: dict[str, Any], *, role: str, scope: str,
    panel: str, metric: str, value: Any, ci_low: Any = None, ci_high: Any = None,
    n: Any = None, sample_id: str = "", organ: str = "", notes: str = "",
    comparison_group: str = "matched_validation_whole_slide_v1",
) -> None:
    """Append one normalized metric row when a finite value exists."""
    value = number(value)
    if value is None:
        return
    rows.append({
        **base,
        "prediction_role": role,
        "scope": scope,
        "sample_id": sample_id,
        "organ": organ,
        "panel": panel,
        "metric": metric,
        "value": value,
        "ci_low": number(ci_low),
        "ci_high": number(ci_high),
        "n": number(n),
        "higher_is_better": direction(metric),
        "comparison_group": comparison_group,
        "notes": notes,
    })


def rows_complete(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from the comprehensive SpatEX evaluator."""
    rows: list[dict[str, Any]] = []
    base = metadata(label, report)
    for role, panels in report.get("aggregate_by_slide", {}).items():
        for panel, metrics in panels.items():
            for metric, summary in metrics.items():
                append_row(
                    rows, base, role=role, scope="validation_macro_slide",
                    panel=panel, metric=metric, value=summary.get("mean"),
                    ci_low=summary.get("ci_low"), ci_high=summary.get("ci_high"),
                    n=summary.get("n"),
                    notes="target-informed control" if role in {
                        "posterior_reconstruction", "shuffled_posterior"
                    } else "",
                )
    for slide in report.get("slides", []):
        for role, path in slide.get("paths", {}).items():
            for panel, payload in path.get("panels", {}).items():
                metrics = {
                    **flatten(payload.get("point", {})),
                    **flatten(payload.get("structured", {}), "structured"),
                }
                for metric, value in metrics.items():
                    append_row(
                        rows, base, role=role, scope="validation_slide",
                        sample_id=slide["sample_id"], organ=slide.get("organ", ""),
                        panel=panel, metric=metric, value=value,
                    )
            for metric, value in flatten(path.get("diagnostics", {}), "diagnostic").items():
                append_row(
                    rows, base, role=role, scope="validation_slide",
                    sample_id=slide["sample_id"], organ=slide.get("organ", ""),
                    panel="diagnostic_only", metric=metric, value=value,
                )
        for metric, value in flatten(slide.get("wae_uncertainty", {}), "wae").items():
            append_row(
                rows, base, role=slide.get("primary_path", "predictive_mean"),
                scope="validation_slide", sample_id=slide["sample_id"],
                organ=slide.get("organ", ""), panel="diagnostic_only",
                metric=metric, value=value,
            )
    return rows


def patient_summary(value: Any) -> tuple[Any, Any, Any, Any]:
    """Unpack the legacy patient-aggregated metric format."""
    if isinstance(value, dict):
        return (
            value.get("patient_mean"), value.get("patient_ci95_low"),
            value.get("patient_ci95_high"), value.get("n_patients"),
        )
    return value, None, None, None


def rows_legacy(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from a legacy parallel-gated supervisor report."""
    rows: list[dict[str, Any]] = []
    base = metadata(label, report)
    role = report.get("prediction_roles", {}).get("primary_point_prediction", "model")
    whole = report.get("whole_slide_structured_field_evaluation") or {}
    for panel, metrics in whole.get("point_metrics_patient_aggregated", {}).items():
        for metric, raw in metrics.items():
            value, low, high, n = patient_summary(raw)
            append_row(
                rows, base, role=role, scope="validation_macro_patient",
                panel=panel, metric=metric, value=value, ci_low=low, ci_high=high,
                n=n, comparison_group="legacy_validation_whole_slide_v1",
                notes="legacy panel membership; compare all_genes directly",
            )
    for panel, metrics in whole.get("structured_metrics_patient_aggregated", {}).items():
        for metric, raw in metrics.items():
            value, low, high, n = patient_summary(raw)
            append_row(
                rows, base, role=role, scope="validation_macro_patient",
                panel=panel, metric=f"structured.{metric}", value=value,
                ci_low=low, ci_high=high, n=n,
                comparison_group="legacy_validation_whole_slide_v1",
                notes="legacy panel membership; compare all_genes directly",
            )
    for slide in whole.get("per_slide_records", []):
        for panel, metrics in slide.get("point_metrics", {}).items():
            for metric, value in flatten(metrics).items():
                append_row(
                    rows, base, role=role, scope="validation_slide",
                    sample_id=slide["sample_id"], organ=slide.get("organ", ""),
                    panel=panel, metric=metric, value=value,
                    comparison_group="legacy_validation_whole_slide_v1",
                )
        for panel, metrics in slide.get("structured_field", {}).items():
            for metric, value in flatten(metrics, "structured").items():
                append_row(
                    rows, base, role=role, scope="validation_slide",
                    sample_id=slide["sample_id"], organ=slide.get("organ", ""),
                    panel=panel, metric=metric, value=value,
                    comparison_group="legacy_validation_whole_slide_v1",
                )
    return rows


def rows_stpath(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from a zero-shot STPath supervisor report."""
    rows: list[dict[str, Any]] = []
    base = metadata(label, report)
    role = "pretrained_zero_shot"
    whole = report.get("whole_slide_point_evaluation") or {}
    for panel, metrics in whole.get("point_metrics_patient_aggregated", {}).items():
        for metric, raw in metrics.items():
            value, low, high, n = patient_summary(raw)
            append_row(
                rows, base, role=role, scope="validation_macro_patient",
                panel=panel, metric=metric, value=value, ci_low=low, ci_high=high,
                n=n, notes="pretraining overlaps HEST-1K; contextual baseline",
                comparison_group="stpath_contextual_legacy_spot_universe_v1",
            )
    for slide in whole.get("per_slide_records", []):
        for panel, metrics in slide.get("point_metrics", {}).items():
            for metric, value in flatten(metrics).items():
                append_row(
                    rows, base, role=role, scope="validation_slide",
                    sample_id=slide["sample_id"], organ=slide.get("organ", ""),
                    panel=panel, metric=metric, value=value,
                    notes="pretraining overlaps HEST-1K; contextual baseline",
                    comparison_group="stpath_contextual_legacy_spot_universe_v1",
                )
    return rows


def report_rows(label: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Dispatch one supported report kind."""
    report = json.loads(path.read_text())
    report["_path"] = str(path.resolve())
    kind = report.get("kind")
    if kind == "spatex_complete_evaluation":
        return report, rows_complete(label, report)
    if kind == "conditional_wae_supervisor_evaluation":
        return report, rows_legacy(label, report)
    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        return report, rows_stpath(label, report)
    raise ValueError(f"{path}: unsupported report kind {kind!r}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    """Write a UTF-8 CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wide_rows(long_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Pivot macro metrics into one row per model/path/panel."""
    selected = [row for row in long_rows if row["scope"].startswith("validation_macro")]
    identifiers = (
        "model", "model_family", "seed", "checkpoint_step", "prediction_role",
        "scope", "panel", "comparison_group", "expression_space", "source_report", "notes",
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    metric_names = sorted({row["metric"] for row in selected})
    for row in selected:
        key = tuple(row[name] for name in identifiers)
        target = grouped.setdefault(key, {name: row[name] for name in identifiers})
        target[row["metric"]] = row["value"]
        if row["ci_low"] is not None:
            target[f"{row['metric']}__ci_low"] = row["ci_low"]
            target[f"{row['metric']}__ci_high"] = row["ci_high"]
    value_fields = []
    for metric in metric_names:
        value_fields.extend((metric, f"{metric}__ci_low", f"{metric}__ci_high"))
    return list(grouped.values()), [*identifiers, *value_fields]


def per_gene_summary(
    label: str, report: dict[str, Any], output_rows: list[dict[str, Any]],
) -> None:
    """Append one across-slide row per gene from a diagnostics sidecar."""
    raw_path = report.get("per_gene_file")
    if not raw_path:
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        raw_path = whole.get("per_gene_diagnostics_path")
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_file():
        candidate = Path(report["_path"]).parent / path.name
        if candidate.is_file():
            path = candidate
        else:
            return
    with np.load(path, allow_pickle=False) as payload:
        genes = payload["gene_names"].astype(str)
        reserved = {
            "gene_names", "sample_ids", "patient_ids", "organs", "metadata_json",
            "primary_path",
        }
        metrics = [name for name in payload.files if name not in reserved]
        summaries: dict[str, np.ndarray] = {}
        for metric in metrics:
            values = np.asarray(payload[metric], dtype=np.float64)
            if values.ndim == 2 and values.shape[1] == len(genes):
                finite = np.isfinite(values)
                totals = np.where(finite, values, 0.0).sum(axis=0)
                counts = finite.sum(axis=0)
                summaries[metric] = np.divide(
                    totals,
                    counts,
                    out=np.full(len(genes), np.nan, dtype=np.float64),
                    where=counts > 0,
                )
        for index, gene in enumerate(genes):
            row: dict[str, Any] = {"model": label, "gene": gene}
            for metric, values in summaries.items():
                row[metric] = float(values[index]) if np.isfinite(values[index]) else None
            output_rows.append(row)


PRESENTATION_METRICS = {
    "pcc": ("pcc",),
    "pcc_ci_low": ("pcc__ci_low",),
    "pcc_ci_high": ("pcc__ci_high",),
    "spearman": ("spearman",),
    "rmse": ("rmse",),
    "mae": ("mae",),
    "auc": ("auc",),
    "median_gene_pcc": ("median_gene_pcc",),
    "fraction_gene_pcc_gt_0_1": ("fraction_gene_pcc_gt_0_1",),
    "spot_profile_pcc": (
        "structured.spot_profile.mean_spot_profile_pcc",
        "structured.spot_profile.pcc",
    ),
    "coexpression_pcc": (
        "structured.coexpression.pcc",
        "structured.coexpression.correlation_matrix_pcc",
    ),
    "moran_pcc": (
        "structured.moran.pcc",
        "structured.moran_local.moran_i_pcc",
    ),
    "local_gradient_pcc": (
        "structured.gradient_local.pcc",
        "structured.gradient_local.signed_gradient_pcc",
    ),
    "wide_gradient_pcc": (
        "structured.gradient_wide.pcc",
        "structured.gradient_wide.signed_gradient_pcc",
    ),
    "local_energy_ratio": ("structured.gradient_local.energy_ratio",),
    "wide_energy_ratio": ("structured.gradient_wide.energy_ratio",),
    "local_sign_agreement": ("structured.gradient_local.sign_agreement",),
    "wide_sign_agreement": ("structured.gradient_wide.sign_agreement",),
    "spatial_ssim": (
        "structured.spatial_ssim.mean",
        "structured.spatial_ssim.mean_per_gene_ssim",
    ),
}

DIAGNOSTIC_METRICS = {
    "amplitude_std_ratio": "diagnostic.amplitude.median_std_ratio",
    "effective_rank_ratio": "diagnostic.template_reuse.entropy_effective_rank_ratio",
    "prediction_effective_rank": "diagnostic.template_reuse.predicted_entropy_effective_rank",
    "target_effective_rank": "diagnostic.template_reuse.target_entropy_effective_rank",
    "exact_pcc": "diagnostic.spatial_resolution.exact_pcc",
    "blur1_pcc": "diagnostic.spatial_resolution.blur1_pcc",
    "blur2_pcc": "diagnostic.spatial_resolution.blur2_pcc",
    "blur1_gain": "diagnostic.spatial_resolution.blur1_gain",
    "blur2_gain": "diagnostic.spatial_resolution.blur2_gain",
    "template_top1": "diagnostic.template_reuse.same_gene_top1_fraction",
    "template_top5": "diagnostic.template_reuse.same_gene_top5_fraction",
    "template_median_rank": "diagnostic.template_reuse.median_same_gene_rank",
    "template_max_reuse": "diagnostic.template_reuse.maximum_target_reuse_count",
    "template_unique_fraction": "diagnostic.template_reuse.unique_top_target_fraction",
    "wae_pairwise_draw_pcc": "wae.mean_pairwise_draw_pcc",
    "wae_pairwise_draw_rmse": "wae.mean_pairwise_draw_rmse",
    "wae_predictive_std": "wae.mean_predictive_std",
    "wae_std_to_target": "wae.predictive_std_to_target_std_ratio",
    "wae_uncertainty_error_pcc": "wae.uncertainty_error_pcc",
}


def canonical_panel(panel: str) -> str:
    """Map historical panel names to the current presentation labels."""
    aliases = {
        "train_log1p_variance_top50": "HVG-50",
        "train_log1p_variance_top200": "HVG-200",
        "train_within_slide_variance_top50": "within-50",
        "train_within_slide_variance_top200": "within-200",
    }
    return aliases.get(panel, panel)


def primary_roles(inventory: list[dict[str, Any]]) -> dict[str, str]:
    """Return the primary prediction path selected by each report."""
    return {row["model"]: row["primary_path"] for row in inventory}


def presentation_rows(
    long_rows: list[dict[str, Any]], inventory: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a compact table with canonical metric columns."""
    roles = primary_roles(inventory)
    diagnostic: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    grouped_diag: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in long_rows:
        if row["prediction_role"] != roles.get(row["model"]):
            continue
        if row["scope"] == "validation_slide" and row["panel"] == "diagnostic_only":
            grouped_diag[(row["model"], row["comparison_group"], row["metric"])].append(row["value"])
    for (model, comparison, metric), values in grouped_diag.items():
        diagnostic[(model, comparison)][metric] = float(np.mean(values))

    macro: dict[tuple[Any, ...], dict[str, Any]] = {}
    identifiers = (
        "model", "model_family", "seed", "checkpoint_step", "prediction_role",
        "comparison_group", "expression_space", "source_report",
    )
    for row in long_rows:
        if not row["scope"].startswith("validation_macro"):
            continue
        if row["prediction_role"] != roles.get(row["model"]):
            continue
        panel = canonical_panel(row["panel"])
        key = (*[row[name] for name in identifiers], panel)
        target = macro.setdefault(
            key, {**{name: row[name] for name in identifiers}, "panel": panel},
        )
        target[row["metric"]] = row["value"]
        if row["ci_low"] is not None:
            target[f"{row['metric']}__ci_low"] = row["ci_low"]
            target[f"{row['metric']}__ci_high"] = row["ci_high"]

    output: list[dict[str, Any]] = []
    for row in macro.values():
        compact = {name: row.get(name) for name in (*identifiers, "panel")}
        for output_name, candidates in PRESENTATION_METRICS.items():
            compact[output_name] = next(
                (row[name] for name in candidates if row.get(name) is not None), None,
            )
        diag = diagnostic.get((row["model"], row["comparison_group"]), {})
        for output_name, source_name in DIAGNOSTIC_METRICS.items():
            compact[output_name] = diag.get(source_name) if row["panel"] == "all_genes" else None
        output.append(compact)
    output.sort(key=lambda row: (row["model"], row["panel"]))
    fields = [
        *identifiers, "panel", *PRESENTATION_METRICS, *DIAGNOSTIC_METRICS,
    ]
    return output, fields


def seed_variability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repeated seeds without mixing architectures or panels."""
    numeric = [*PRESENTATION_METRICS, *DIAGNOSTIC_METRICS]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        experiment = re.sub(r"_seed(?:1|2|10)$", "", row["model"])
        groups[(experiment, row["panel"], row["comparison_group"])].append(row)
    output = []
    for (experiment, panel, comparison), members in groups.items():
        seeds = {row["seed"] for row in members if row["seed"] is not None}
        if len(seeds) < 2:
            continue
        for metric in numeric:
            values = [number(row.get(metric)) for row in members]
            values = [value for value in values if value is not None]
            if len(values) < 2:
                continue
            output.append({
                "experiment": experiment,
                "panel": panel,
                "comparison_group": comparison,
                "metric": metric,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": min(values),
                "maximum": max(values),
                "n_seeds": len(values),
                "seeds": ";".join(str(seed) for seed in sorted(seeds)),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="label=/path/report.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    per_gene_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for value in args.report:
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        report, rows = report_rows(label, path)
        long_rows.extend(rows)
        per_gene_summary(label, report, per_gene_rows)
        meta = metadata(label, report)
        inventory.append({
            **meta,
            "report_kind": report.get("kind"),
            "primary_path": report.get("primary_path") or report.get("prediction_roles", {}).get("primary_point_prediction") or "pretrained_zero_shot",
            "split": report.get("split"),
            "n_metric_rows": len(rows),
        })

    long_rows.sort(key=lambda row: tuple(str(row[name]) for name in (
        "model", "prediction_role", "scope", "sample_id", "panel", "metric",
    )))
    write_csv(output / "metrics_long.csv", long_rows, BASE_FIELDS)
    wide, wide_fields = wide_rows(long_rows)
    write_csv(output / "metrics_wide_summary.csv", wide, wide_fields)
    write_csv(output / "model_inventory.csv", inventory, list(inventory[0]))
    presentation, presentation_fields = presentation_rows(long_rows, inventory)
    write_csv(output / "presentation_summary.csv", presentation, presentation_fields)
    variability = seed_variability_rows(presentation)
    if variability:
        write_csv(output / "seed_variability.csv", variability, list(variability[0]))

    if per_gene_rows:
        fields = ["model", "gene", *sorted({key for row in per_gene_rows for key in row if key not in {"model", "gene"}})]
        with gzip.open(output / "per_gene_summary.csv.gz", "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_gene_rows)

    print(json.dumps({
        "reports": len(args.report),
        "metric_rows": len(long_rows),
        "wide_rows": len(wide),
        "presentation_rows": len(presentation),
        "seed_variability_rows": len(variability),
        "per_gene_rows": len(per_gene_rows),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
