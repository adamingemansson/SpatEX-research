"""Fit the gene-program basis used for within-spot refinement.

Expression is centered within each training slide before balanced PCA so that
the basis emphasizes within-tissue structure rather than slide offsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class CenteredGeneStructure:
    """Training-derived gene basis, mean, scale, and fixed gene order."""
    basis: torch.Tensor
    global_mean: torch.Tensor
    per_gene_scale: torch.Tensor
    gene_names: tuple[str, ...]

    def validate(self) -> "CenteredGeneStructure":
        """Validate dimensions, finite values, positive scales, and gene names."""
        rank, genes = self.basis.shape
        if rank < 1 or genes < 1:
            raise ValueError("empty gene structure")
        if self.global_mean.shape != (genes,) or self.per_gene_scale.shape != (genes,):
            raise ValueError("gene structure tensors have incompatible shapes")
        if len(self.gene_names) != genes:
            raise ValueError("gene name count does not match structure")
        if not bool(torch.isfinite(self.basis).all()):
            raise ValueError("basis contains non-finite values")
        if not bool(torch.isfinite(self.global_mean).all()):
            raise ValueError("global mean contains non-finite values")
        if not bool(torch.isfinite(self.per_gene_scale).all()):
            raise ValueError("per-gene scale contains non-finite values")
        if not bool((self.per_gene_scale > 0).all()):
            raise ValueError("per-gene scale must be positive")
        return self

    def to(self, device: torch.device | str) -> "CenteredGeneStructure":
        """Return the structure tensors on another device."""
        return CenteredGeneStructure(
            basis=self.basis.to(device),
            global_mean=self.global_mean.to(device),
            per_gene_scale=self.per_gene_scale.to(device),
            gene_names=self.gene_names,
        )


def gene_hash(gene_names: tuple[str, ...] | list[str]) -> str:
    """Hash an ordered gene list for checkpoint compatibility checks."""
    return hashlib.sha256("\n".join(gene_names).encode()).hexdigest()


def save_structure(structure: CenteredGeneStructure, path: str | Path) -> None:
    """Save a validated gene structure as a portable Torch payload."""
    structure.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "basis": structure.basis.cpu(),
            "global_mean": structure.global_mean.cpu(),
            "per_gene_scale": structure.per_gene_scale.cpu(),
            "gene_names": list(structure.gene_names),
            "gene_hash": gene_hash(structure.gene_names),
        },
        path,
    )


def load_structure(path: str | Path) -> CenteredGeneStructure:
    """Load a gene structure and verify its stored gene-order hash."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    names = tuple(str(name) for name in payload["gene_names"])
    if payload.get("gene_hash") != gene_hash(names):
        raise ValueError(f"{path}: gene hash mismatch")
    return CenteredGeneStructure(
        basis=payload["basis"].float(),
        global_mean=payload["global_mean"].float(),
        per_gene_scale=payload["per_gene_scale"].float(),
        gene_names=names,
    ).validate()


def fit_centered_structure(
    slide_arrays: list[np.ndarray],
    gene_names: list[str],
    rank: int,
    seed: int,
    organs: list[str] | None = None,
) -> CenteredGeneStructure:
    """Fit a per-slide-centered PCA basis with balanced organ weighting."""
    if not slide_arrays:
        raise ValueError("at least one training slide is required")
    genes = len(gene_names)
    if organs is None:
        organs = [f"slide_{index}" for index in range(len(slide_arrays))]
    if len(organs) != len(slide_arrays) or any(not organ for organ in organs):
        raise ValueError("one non-empty organ label is required per slide")
    slides_per_organ = {organ: organs.count(organ) for organ in set(organs)}
    centered_by_slide: list[np.ndarray] = []
    means_by_organ: dict[str, list[np.ndarray]] = {}
    moments_by_organ: dict[str, list[np.ndarray]] = {}
    for array, organ in zip(slide_arrays, organs):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != genes:
            raise ValueError("training expression has incompatible gene dimension")
        slide_mean = array.mean(axis=0)
        residual = array - slide_mean[None, :]
        centered_by_slide.append(residual)
        means_by_organ.setdefault(organ, []).append(slide_mean)
        moments_by_organ.setdefault(organ, []).append(
            np.mean(np.square(residual, dtype=np.float64), axis=0)
        )

    # Estimate global statistics without letting large organs dominate.
    global_mean = np.mean(
        [np.mean(values, axis=0) for values in means_by_organ.values()], axis=0
    ).astype(np.float32)
    scale = np.sqrt(
        np.mean(
            [np.mean(values, axis=0) for values in moments_by_organ.values()], axis=0
        )
    )
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    weighted = []
    for residual, organ in zip(centered_by_slide, organs):
        standardized = residual / scale[None, :]
        # Give each slide equal weight within its organ.
        standardized *= np.float32(
            1.0 / np.sqrt(len(residual) * slides_per_organ[organ])
        )
        weighted.append(standardized)
    standardized = np.concatenate(weighted, axis=0).astype(np.float32)
    torch.manual_seed(seed)
    tensor = torch.from_numpy(standardized)
    q = min(int(rank), min(tensor.shape))
    _, _, right = torch.pca_lowrank(tensor, q=q, center=False)
    return CenteredGeneStructure(
        basis=right.T.contiguous().float(),
        global_mean=torch.from_numpy(global_mean).float(),
        per_gene_scale=torch.from_numpy(scale).float(),
        gene_names=tuple(gene_names),
    ).validate()


def main() -> None:
    """Fit the training-only gene structure from the command line."""
    parser = argparse.ArgumentParser(description="Fit training-only centered gene structure")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    names = list(manifest["gene_names"])
    arrays = []
    organs = []
    for slide in manifest["slides"]:
        if slide["split"] != "train":
            continue
        path = Path(slide["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with np.load(path, allow_pickle=False) as payload:
            arrays.append(payload["expression"].astype(np.float32))
        organs.append(str(slide.get("organ", "unknown")))
    structure = fit_centered_structure(arrays, names, args.rank, args.seed, organs)
    save_structure(structure, args.output)
    print(f"saved centered gene structure {tuple(structure.basis.shape)} to {args.output}")


if __name__ == "__main__":
    main()
