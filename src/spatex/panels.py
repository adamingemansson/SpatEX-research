"""Fit train-derived gene panels for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatex.data import PreparedManifest, SlideRecord


def fit_panels(manifest_path: str | Path, output: str | Path) -> Path:
    """Fit global- and within-slide-variance panels using training slides only."""
    manifest = PreparedManifest.load(manifest_path)
    global_sum = np.zeros(len(manifest.gene_names), dtype=np.float64)
    global_square = np.zeros_like(global_sum)
    within_variance = np.zeros_like(global_sum)
    total_rows = 0
    slides = manifest.for_split("train")
    if not slides:
        raise ValueError("training slides are required to fit panels")
    for index, spec in enumerate(slides, 1):
        expression = SlideRecord.load(spec, len(manifest.gene_names)).expression.astype(
            np.float64, copy=False
        )
        global_sum += expression.sum(axis=0)
        global_square += np.square(expression).sum(axis=0)
        within_variance += expression.var(axis=0)
        total_rows += len(expression)
        print(f"panel fitting {index}/{len(slides)}: {spec.sample_id}", flush=True)
    global_variance = global_square / total_rows - np.square(global_sum / total_rows)
    within_variance /= len(slides)

    def top(values: np.ndarray, n: int) -> list[int]:
        return np.argsort(values, kind="stable")[-n:][::-1].astype(int).tolist()

    panels = {
        "all_genes": list(range(len(manifest.gene_names))),
        "HVG-50": top(global_variance, 50),
        "HVG-200": top(global_variance, 200),
        "within-50": top(within_variance, 50),
        "within-200": top(within_variance, 200),
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "source_manifest": str(Path(manifest_path).expanduser().resolve()),
                "fit_split": "train",
                "gene_names": list(manifest.gene_names),
                "panels": {
                    name: {
                        "indices": indices,
                        "genes": [manifest.gene_names[index] for index in indices],
                    }
                    for name, indices in panels.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"train-derived panels saved to {output}")
    return output


def load_panels(
    path: str | Path | None, gene_names: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    """Load a compatible panel artifact or return the all-gene panel."""
    if path is None:
        return {"all_genes": tuple(range(len(gene_names)))}
    payload = json.loads(Path(path).expanduser().read_text())
    if tuple(payload["gene_names"]) != gene_names:
        raise ValueError("panel artifact gene order does not match the prepared data")
    return {
        str(name): tuple(int(index) for index in values["indices"])
        for name, values in payload["panels"].items()
    }


def main() -> None:
    """Fit train-derived evaluation panels."""
    parser = argparse.ArgumentParser(description="Fit train-derived SpatEX gene panels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_panels(args.manifest, args.output)


if __name__ == "__main__":
    main()
