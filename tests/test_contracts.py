"""Tests for configuration, coordinates, and gene-structure contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from spatex.config import load_config
from spatex.graph import normalize_coordinates
from spatex.structure import fit_centered_structure, load_structure, save_structure


def test_recursive_config_inheritance(tmp_path: Path):
    """Child configurations should override nested parent values."""
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"model": {"context_dim": 8, "dropout": 0.2}, "seed": 10})
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"inherits": "base.yaml", "model": {"dropout": 0.0}})
    )
    config = load_config(tmp_path / "child.yaml")
    assert config == {"model": {"context_dim": 8, "dropout": 0.0}, "seed": 10}


def test_coordinate_normalization_uses_spot_spacing():
    """Normalized adjacent spots should be one spacing unit apart."""
    coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    normalized = normalize_coordinates(coordinates)
    assert np.allclose(normalized.mean(axis=0), 0.0)
    assert np.allclose(np.diff(normalized[:, 0]), 1.0)


def test_centered_structure_round_trip(tmp_path: Path):
    """A fitted gene structure should survive a save-load round trip."""
    rng = np.random.default_rng(3)
    slides = [rng.normal(size=(20, 7)).astype(np.float32) for _ in range(3)]
    names = [f"g{index}" for index in range(7)]
    structure = fit_centered_structure(slides, names, rank=3, seed=0)
    path = tmp_path / "structure.pt"
    save_structure(structure, path)
    loaded = load_structure(path)
    assert loaded.gene_names == tuple(names)
    torch.testing.assert_close(loaded.basis, structure.basis)
    assert loaded.basis.shape == (3, 7)
