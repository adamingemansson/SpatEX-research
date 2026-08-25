"""Prepare aligned SpatEX slide records.

HEST expression and cached UNI2 features are joined by barcode, normalized,
and stored in a compact per-slide format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from spatex.graph import normalize_coordinates


def _sample_ids(manifest: dict[str, Any]) -> dict[str, str]:
    """Extract sample-to-split assignments from supported manifest layouts."""
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for sample_id in manifest.get(f"{split}_sample_ids", []):
            result[str(sample_id)] = split
    if not result and isinstance(manifest.get("samples"), list):
        for sample in manifest["samples"]:
            result[str(sample["sample_id"])] = str(sample["split"])
    if not result:
        raise ValueError("source manifest has no recognizable sample splits")
    return result


def _gene_names(manifest: dict[str, Any]) -> list[str]:
    """Read the ordered output gene panel from a source manifest."""
    for key in ("gene_panel", "gene_names", "genes"):
        value = manifest.get(key)
        if isinstance(value, list) and value:
            return [str(gene) for gene in value]
    raise ValueError("source manifest has no recognizable ordered gene panel")


def _organ(manifest: dict[str, Any], sample_id: str) -> str:
    """Resolve an organ label from supported manifest layouts."""

    def label(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("organ", "organ_type", "tissue", "tissue_type"):
                if value.get(key) is not None:
                    return str(value[key])
        return None

    for key in ("sample_to_organ", "organ_by_sample", "sample_organs"):
        mapping = manifest.get(key)
        if isinstance(mapping, dict) and sample_id in mapping:
            result = label(mapping[sample_id])
            if result is not None:
                return result

    samples = manifest.get("samples", {})

    if isinstance(samples, dict):
        result = label(samples.get(sample_id))
        if result is not None:
            return result

    elif isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if str(sample.get("sample_id")) == sample_id:
                return label(sample) or "unknown"

    return "unknown"


def _matrix(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Select genes and create normalize-total log1p expression targets."""
    index = {str(name): i for i, name in enumerate(adata.var_names)}
    missing = [gene for gene in genes if gene not in index]
    if missing:
        raise ValueError(f"h5ad lacks {len(missing)} manifest genes; first={missing[:5]}")
    columns = [index[gene] for gene in genes]
    matrix = adata.X[:, columns]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    # Normalize each spot before applying log1p.
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix * (10000.0 / np.maximum(totals, 1.0))
    return np.log1p(normalized).astype(np.float32)


def _cache(cache_dir: Path, sample_id: str) -> dict[str, np.ndarray]:
    """Load cached UNI2 features, barcodes, and patch availability."""
    path = cache_dir / f"{sample_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing UNI2 cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        barcodes = payload["barcodes"].astype(str)
        available = (
            payload["image_source_available"].astype(bool)
            if "image_source_available" in payload
            else np.ones(len(features), dtype=bool)
        )
    return {"features": features, "barcodes": barcodes, "available": available}


def prepare(
    source_manifest: str | Path,
    hest_root: str | Path,
    uni2_cache: str | Path,
    output_root: str | Path,
) -> Path:
    """Prepare aligned per-slide NPZ files and their compact manifest."""
    source = json.loads(Path(source_manifest).read_text())
    splits = _sample_ids(source)
    genes = _gene_names(source)
    hest_root = Path(hest_root).expanduser().resolve()
    cache_dir = Path(uni2_cache).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    records_dir = output_root / "slides"
    records_dir.mkdir(parents=True, exist_ok=True)
    slides = []
    for item, (sample_id, split) in enumerate(sorted(splits.items()), 1):
        h5ad_path = hest_root / "st" / f"{sample_id}.h5ad"
        if not h5ad_path.is_file():
            candidates = list(hest_root.rglob(f"{sample_id}.h5ad"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"no unambiguous h5ad for {sample_id}")
            h5ad_path = candidates[0]
        adata = ad.read_h5ad(h5ad_path)
        cache = _cache(cache_dir, sample_id)
        adata_barcodes = np.asarray(adata.obs_names, dtype=str)

        # Reorder cached features and retain spots without image patches.
        cache_index = {barcode: index for index, barcode in enumerate(cache["barcodes"])}
        if len(cache_index) != len(cache["barcodes"]):
            raise ValueError(f"{sample_id}: UNI2 cache contains duplicate barcodes")

        order = np.asarray(
            [cache_index.get(barcode, -1) for barcode in adata_barcodes],
            dtype=np.int64,
        )
        present = order >= 0

        aligned_features = np.zeros(
            (len(adata_barcodes), cache["features"].shape[1]),
            dtype=np.float32,
        )
        aligned_available = np.zeros(len(adata_barcodes), dtype=bool)
        aligned_features[present] = cache["features"][order[present]]
        aligned_available[present] = cache["available"][order[present]]

        missing_count = int((~present).sum())
        if missing_count:
            print(
                f"{sample_id}: retaining {missing_count} spots without image patches",
                flush=True,
            )
        coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        output = records_dir / f"{sample_id}.npz"
        np.savez_compressed(
            output,
            image_features=cache["features"][order],
            coordinates=normalize_coordinates(coordinates),
            image_available=cache["available"][order],
            expression=_matrix(adata, genes),
            barcodes=adata_barcodes,
        )
        slides.append(
            {
                "sample_id": sample_id,
                "split": split,
                "organ": _organ(source, sample_id),
                "path": str(output.relative_to(output_root)),
            }
        )
        print(f"prepared {item}/{len(splits)} {sample_id}: {len(adata)} spots", flush=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "spatex_prepared_manifest",
                "version": 1,
                "gene_names": genes,
                "slides": slides,
                "normalization": "normalize_total_10000_then_log1p",
                "coordinate_normalization": "center_then_median_nearest_neighbor_spacing",
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path


def main() -> None:
    """Prepare a dataset from the command line."""
    parser = argparse.ArgumentParser(description="Prepare SpatEX slide records")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--hest-root", required=True)
    parser.add_argument("--uni2-cache", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    path = prepare(args.source_manifest, args.hest_root, args.uni2_cache, args.output_root)
    print(f"prepared manifest saved to {path}")


if __name__ == "__main__":
    main()
