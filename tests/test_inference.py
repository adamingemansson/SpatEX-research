"""Tests for the expression-free inference input boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from spatex.infer import load_inference_slide


def test_inference_slide_requires_no_expression(tmp_path: Path):
    """A valid inference slide should contain image and spatial inputs only."""
    path = tmp_path / "slide.npz"
    np.savez_compressed(
        path,
        image_features=np.zeros((5, 12), dtype=np.float32),
        coordinates=np.zeros((5, 2), dtype=np.float32),
        image_available=np.ones(5, dtype=bool),
        barcodes=np.asarray([f"b{i}" for i in range(5)]),
    )
    inputs, barcodes = load_inference_slide(path, torch.device("cpu"))
    assert inputs.n_queries == 5
    assert list(barcodes) == [f"b{i}" for i in range(5)]


def test_inference_slide_fails_closed_on_missing_field(tmp_path: Path):
    """Missing required inference fields should raise a clear error."""
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, image_features=np.zeros((2, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="missing inference fields"):
        load_inference_slide(path, torch.device("cpu"))
