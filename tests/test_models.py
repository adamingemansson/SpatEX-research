"""Shape, leakage, and stochastic-path tests for all model variants."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from spatex.inputs import InputBatch
from spatex.models.factory import build_model
from spatex.models.wae import SpatEXWAE
from spatex.structure import CenteredGeneStructure


def config(kind: str, prior: str = "standard", conditioning: str = "none") -> dict:
    """Build a small model configuration for unit tests."""
    return {
        "model": {
            "kind": kind,
            "prior": prior,
            "image_feature_dim": 12,
            "context_dim": 16,
            "image_proj_dim": 8,
            "coord_dim": 8,
            "n_heads": 4,
            "n_blocks": 2,
            "dense_threshold": 8,
            "spatial_k": 4,
            "decoder_hidden_dim": 20,
            "dropout": 0.0,
            "gene_structure_hidden_dim": 5,
            "refinement_k": 3,
            "refinement_hidden_dim": 7,
            "refinement_gex_dim": 6,
            "latent_dim": 4,
            "wae_hidden_dim": 18,
            "mmd_weight": 0.1,
            "deterministic_weight": 1.0,
            "residual_mode": "antithetic_zero_mean",
            "posterior_conditioning": conditioning,
            "freeze_deterministic_backbone": False,
        }
    }


@pytest.fixture
def structure() -> CenteredGeneStructure:
    """Create a compact valid gene basis."""
    torch.manual_seed(4)
    return CenteredGeneStructure(
        basis=torch.randn(3, 9),
        global_mean=torch.randn(9),
        per_gene_scale=torch.rand(9) + 0.2,
        gene_names=tuple(f"gene_{index}" for index in range(9)),
    ).validate()


@pytest.fixture
def inputs() -> InputBatch:
    """Create mixed-availability inputs with a partial query mask."""
    torch.manual_seed(5)
    return InputBatch(
        sample_id="slide",
        image_features=torch.randn(11, 12),
        coordinates=torch.randn(11, 2),
        image_available=torch.tensor([True] * 10 + [False]),
        query_mask=torch.tensor(
            [True, False, True, True, False, True, True, False, True, True, True]
        ),
    ).validate(12)


def test_deployable_input_has_no_expression_field():
    """Measured expression must not be representable as deployable input."""
    assert {field.name for field in fields(InputBatch)} == {
        "sample_id",
        "image_features",
        "coordinates",
        "image_available",
        "query_mask",
    }


def test_parallel_gated_deterministic_forward(structure, inputs):
    """The deterministic model should expose valid gated refinement paths."""
    model = build_model(config("deterministic"), structure).eval()
    result = model.predict_all(inputs)
    assert result["expression"].shape == (inputs.n_queries, 9)
    assert result["context"].shape == (inputs.n_queries, 16)
    assert result["gates"].shape == (2,)
    assert torch.isfinite(result["expression"]).all()
    result["expression"].sum().backward()
    assert model.composition_logits.grad is not None


@pytest.mark.parametrize(
    ("prior", "conditioning"),
    [("standard", "none"), ("conditional", "film")],
)
def test_wae_paths_are_explicit_and_shape_safe(structure, inputs, prior, conditioning):
    """Both WAE variants should return the documented prediction paths."""
    torch.manual_seed(7)
    model = build_model(config("wae", prior, conditioning), structure).eval()
    assert isinstance(model, SpatEXWAE)
    target = torch.randn(inputs.n_queries, 9)
    sampled = model.sample(inputs, n_draws=8)
    assert sampled["samples"].shape == (8, inputs.n_queries, 9)
    assert sampled["predictive_variance"].shape == target.shape
    assert sampled["context"].shape == (inputs.n_queries, 16)
    posterior = model.posterior_reconstruction(inputs, target)
    assert posterior["prediction"].shape == target.shape
    controls = model.controls(inputs, target)
    assert set(controls) == {
        "posterior_reconstruction",
        "zero_latent",
        "shuffled_posterior",
    }


@pytest.mark.parametrize("prior", ["standard", "conditional"])
def test_antithetic_pair_mean_is_exact_deterministic(structure, inputs, prior):
    """A complete antithetic pair should average to the deterministic output."""
    torch.manual_seed(11)
    conditioning = "film" if prior == "conditional" else "none"
    model = build_model(config("wae", prior, conditioning), structure).eval()
    sampled = model.sample(inputs, n_draws=2)
    torch.testing.assert_close(
        sampled["predictive_mean"], sampled["deterministic"], rtol=1e-6, atol=1e-6
    )


def test_target_only_enters_explicit_training_path(structure, inputs):
    """Calling posterior diagnostics must not change deployable inference."""
    model = build_model(config("wae", "conditional", "film"), structure).eval()
    target_a = torch.randn(inputs.n_queries, 9)
    target_b = torch.randn(inputs.n_queries, 9)
    torch.manual_seed(13)
    inference_a = model.sample(inputs, n_draws=2)["predictive_mean"]
    model.posterior_reconstruction(inputs, target_a)
    model.posterior_reconstruction(inputs, target_b)
    torch.manual_seed(13)
    inference_b = model.sample(inputs, n_draws=2)["predictive_mean"]
    torch.testing.assert_close(inference_a, inference_b)


def test_wae_training_loss_is_finite(structure, inputs):
    """The combined WAE objective should be finite and differentiable."""
    model = build_model(config("wae", "standard", "none"), structure).train()
    target = torch.randn(inputs.n_queries, 9)
    loss, metrics = model.training_loss(inputs, target, 0.1, config("wae"))
    assert torch.isfinite(loss)
    assert {"rmse", "pcc_loss", "deterministic_loss", "mmd", "total"} <= set(metrics)
    loss.backward()
