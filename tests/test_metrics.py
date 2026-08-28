"""Tests for study metrics and train-derived panels."""

from __future__ import annotations

import torch

from spatex.metrics import flattened_pcc, macro_gene_pcc, panel_metrics


def test_gene_wise_pcc_is_not_flattened_pcc():
    """The primary score must not collapse spots and genes into one vector."""
    target = torch.tensor([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]])
    prediction = torch.tensor([[2.0, 10.0], [1.0, 10.0], [0.0, 10.0]])
    assert torch.isclose(macro_gene_pcc(prediction, target), torch.tensor(-1.0))
    assert not torch.isclose(
        flattened_pcc(prediction, target), macro_gene_pcc(prediction, target)
    )


def test_panel_metrics_include_collapse_diagnostics():
    """Evaluation reports point accuracy, amplitude, and expression rank."""
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(12, 8, generator=generator)
    prediction = 0.2 * target
    report, genes = panel_metrics(
        prediction,
        target,
        {"all_genes": tuple(range(8)), "small": (0, 1, 2)},
    )
    assert report["all_genes"]["gene_pcc"] > 0.99
    assert 0.19 < report["all_genes"]["amplitude_ratio"] < 0.21
    assert report["all_genes"]["prediction_effective_rank"] > 0
    assert torch.isnan(torch.tensor(report["small"]["prediction_effective_rank"]))
    assert genes["all_genes"].shape == (8,)
