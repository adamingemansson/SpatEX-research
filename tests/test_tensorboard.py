"""Tests for compact TensorBoard diagnostics."""

import pytest
import torch

from spatex.tensorboard import example_map, spatial_gene_metrics


def test_example_map_skips_unknown_genes_and_deduplicates_indices():
    logging = {
        "spatial_examples": [
            {"sample_id": "slide_a", "genes": ["B", "missing", "B"]},
            {"sample_id": "slide_b", "genes": ["A"]},
        ]
    }
    assert example_map(logging, ("A", "B")) == {
        "slide_a": (1,),
        "slide_b": (0,),
    }


def test_spatial_gene_metrics_match_training_loss_definition():
    target = torch.tensor([0.0, 1.0, 2.0])
    prediction = torch.tensor([0.0, 1.0, 2.0])

    metrics = spatial_gene_metrics(target, prediction, pcc_weight=0.1)

    assert metrics["pcc"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["total"] == pytest.approx(0.0)
    assert metrics["std_ratio"] == pytest.approx(1.0)


def test_spatial_gene_metrics_keep_amplitude_error_visible():
    target = torch.tensor([0.0, 1.0, 2.0])
    prediction = 0.5 * target

    metrics = spatial_gene_metrics(target, prediction, pcc_weight=0.1)

    assert metrics["pcc"] == pytest.approx(1.0)
    assert metrics["rmse"] > 0.0
    assert metrics["total"] == pytest.approx(metrics["rmse"])
    assert metrics["std_ratio"] == pytest.approx(0.5)
