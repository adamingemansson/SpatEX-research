"""Read prepared slides and construct model-ready spatial fields.

Expression targets are returned separately from the H&E-only ``InputBatch``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spatex.inputs import InputBatch


@dataclass(frozen=True)
class SlideSpec:
    """Location and split metadata for one prepared slide."""
    sample_id: str
    split: str
    organ: str
    path: Path


@dataclass(frozen=True)
class PreparedManifest:
    """Ordered genes and slide records produced by data preparation."""
    path: Path
    gene_names: tuple[str, ...]
    slides: tuple[SlideSpec, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PreparedManifest":
        """Load a prepared manifest and resolve relative slide paths."""
        path = Path(path).expanduser().resolve()
        payload = json.loads(path.read_text())
        names = tuple(str(value) for value in payload["gene_names"])
        slides = []
        for value in payload["slides"]:
            record_path = Path(value["path"])
            if not record_path.is_absolute():
                record_path = path.parent / record_path
            slides.append(
                SlideSpec(
                    sample_id=str(value["sample_id"]),
                    split=str(value["split"]),
                    organ=str(value.get("organ", "unknown")),
                    path=record_path.resolve(),
                )
            )
        manifest = cls(path=path, gene_names=names, slides=tuple(slides))
        if not names or not slides:
            raise ValueError(f"{path}: empty prepared manifest")
        return manifest

    def for_split(self, split: str) -> tuple[SlideSpec, ...]:
        """Return slides assigned to one dataset split."""
        return tuple(slide for slide in self.slides if slide.split == split)


@dataclass(frozen=True)
class SlideRecord:
    """Aligned UNI2 features, coordinates, expression, and spot metadata."""
    sample_id: str
    organ: str
    image_features: np.ndarray
    coordinates: np.ndarray
    image_available: np.ndarray
    expression: np.ndarray
    barcodes: np.ndarray

    @classmethod
    def load(cls, spec: SlideSpec, n_genes: int) -> "SlideRecord":
        """Load one slide and validate spot and gene dimensions."""
        with np.load(spec.path, allow_pickle=False) as payload:
            record = cls(
                sample_id=spec.sample_id,
                organ=spec.organ,
                image_features=payload["image_features"].astype(np.float32),
                coordinates=payload["coordinates"].astype(np.float32),
                image_available=payload["image_available"].astype(bool),
                expression=payload["expression"].astype(np.float32),
                barcodes=payload["barcodes"].astype(str),
            )
        n = len(record.expression)
        if record.expression.shape != (n, n_genes):
            raise ValueError(f"{spec.path}: expression shape mismatch")
        if record.image_features.ndim != 2 or len(record.image_features) != n:
            raise ValueError(f"{spec.path}: image feature shape mismatch")
        if record.coordinates.shape != (n, 2):
            raise ValueError(f"{spec.path}: coordinate shape mismatch")
        if record.image_available.shape != (n,) or record.barcodes.shape != (n,):
            raise ValueError(f"{spec.path}: spot metadata shape mismatch")
        return record

    def field(
        self,
        field_size: int,
        generator: np.random.Generator,
        device: torch.device | str,
    ) -> tuple[InputBatch, torch.Tensor]:
        """Sample the nearest spots around a random training location."""
        n = len(self.expression)
        size = min(int(field_size), n)
        center = int(generator.integers(n))
        # Train on a contiguous field around one randomly chosen spot.
        distances = np.sum((self.coordinates - self.coordinates[center]) ** 2, axis=1)
        indices = np.argpartition(distances, size - 1)[:size]
        indices = indices[np.argsort(distances[indices])]
        return self._batch(indices, device)

    def whole_slide(self, device: torch.device | str) -> tuple[InputBatch, torch.Tensor]:
        """Return every spot for whole-slide validation or evaluation."""
        return self._batch(np.arange(len(self.expression)), device)

    def _batch(
        self, indices: np.ndarray, device: torch.device | str
    ) -> tuple[InputBatch, torch.Tensor]:
        """Convert selected rows into inference inputs and separate targets."""
        indices = np.asarray(indices)
        n = len(indices)
        inputs = InputBatch(
            sample_id=self.sample_id,
            image_features=torch.from_numpy(self.image_features[indices]).to(device),
            coordinates=torch.from_numpy(self.coordinates[indices]).to(device),
            image_available=torch.from_numpy(self.image_available[indices]).to(device),
            query_mask=torch.ones(n, dtype=torch.bool, device=device),
        )
        return inputs, torch.from_numpy(self.expression[indices]).to(device)
