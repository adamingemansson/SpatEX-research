"""Add optional WAE-MMD residuals to the deterministic SpatEX output.

Measured expression enters only the training posterior. Deployable samples
come from either a standard or an H&E-conditioned prior.
"""

from __future__ import annotations

import torch
from torch import nn

from spatex.inputs import InputBatch
from spatex.losses import imq_mmd, point_loss
from spatex.models.deterministic import SpatEX


class ExpressionPosterior(nn.Module):
    """Encode measured expression into a training-only latent."""

    def __init__(
        self,
        n_genes: int,
        context_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conditioning: str,
    ) -> None:
        super().__init__()
        if conditioning not in {"none", "film"}:
            raise ValueError(f"unsupported posterior conditioning: {conditioning}")
        self.conditioning = conditioning
        self.input = nn.Linear(n_genes, hidden_dim)
        if conditioning == "film":
            self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Encode measured training targets, optionally conditioned with FiLM."""
        hidden = torch.nn.functional.gelu(self.input(target))
        if self.conditioning == "film":
            scale, shift = self.film(context).chunk(2, dim=-1)
            hidden = hidden * (1.0 + 0.1 * torch.tanh(scale)) + shift
        return self.output(torch.nn.functional.gelu(hidden))


class ConditionalPrior(nn.Module):
    """Predict a diagonal Gaussian prior from H&E context."""

    def __init__(self, context_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the context-dependent prior mean and positive scale."""
        mean, log_scale = self.network(context).chunk(2, dim=-1)
        return mean, torch.exp(log_scale.clamp(-5.0, 3.0))


class SpatEXWAE(nn.Module):
    """Add a WAE-MMD residual to the deterministic prediction."""

    def __init__(self, config: dict, deterministic: SpatEX) -> None:
        super().__init__()
        model = config["model"]
        self.deterministic = deterministic
        self.prior_kind = str(model["prior"])
        if self.prior_kind not in {"standard", "conditional"}:
            raise ValueError("WAE prior must be standard or conditional")
        self.latent_dim = int(model["latent_dim"])
        context_dim = int(model["context_dim"])
        hidden = int(model["wae_hidden_dim"])
        n_genes = len(deterministic.gene_names)
        self.posterior = ExpressionPosterior(
            n_genes,
            context_dim,
            hidden,
            self.latent_dim,
            str(model.get("posterior_conditioning", "none")),
        )
        self.conditional_prior = (
            ConditionalPrior(context_dim, hidden, self.latent_dim)
            if self.prior_kind == "conditional"
            else None
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(context_dim + self.latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_genes),
        )
        self.residual_mode = str(model.get("residual_mode", "antithetic_zero_mean"))
        if self.residual_mode != "antithetic_zero_mean":
            raise ValueError("only antithetic_zero_mean is supported")
        if bool(model.get("freeze_deterministic_backbone", False)):
            for parameter in self.deterministic.parameters():
                parameter.requires_grad_(False)

    def deterministic_prediction(self, inputs: InputBatch) -> dict[str, torch.Tensor]:
        """Run the shared SpatEX backbone and retain its intermediate outputs."""
        return self.deterministic.predict_all(inputs)

    def prior_parameters(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return standard-normal or H&E-conditioned prior parameters."""
        if self.conditional_prior is None:
            return torch.zeros(
                len(context), self.latent_dim, device=context.device, dtype=context.dtype
            ), torch.ones(
                len(context), self.latent_dim, device=context.device, dtype=context.dtype
            )
        return self.conditional_prior(context)

    def residual(self, context: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode an odd residual around the prior center."""
        center, _ = self.prior_parameters(context)
        delta = latent - center
        positive = self.residual_decoder(torch.cat((context, delta), dim=-1))
        negative = self.residual_decoder(torch.cat((context, -delta), dim=-1))
        return 0.5 * (positive - negative)

    def prediction_from_latent(
        self, deterministic: torch.Tensor, context: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Add one latent residual to the deterministic expression."""
        return deterministic + self.residual(context, latent)

    def sample(
        self, inputs: InputBatch, n_draws: int, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Draw deployable H&E-only predictions and summarize their variance."""
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        outputs = self.deterministic_prediction(inputs)
        deterministic = outputs["expression"]
        context = outputs["context"]
        mean, scale = self.prior_parameters(context)
        draws: list[torch.Tensor] = []
        for draw_index in range(n_draws):
            if draw_index % 2 == 0:
                noise = torch.randn(
                    mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
                )
            else:
                # Pair each draw with its reflection around the prior mean.
                noise = -noise
            latent = mean + scale * noise
            draws.append(self.prediction_from_latent(deterministic, context, latent))
        stack = torch.stack(draws)
        return {
            "deterministic": deterministic,
            "context": context,
            "samples": stack,
            "predictive_mean": stack.mean(dim=0),
            "predictive_variance": stack.var(dim=0, unbiased=False),
        }

    def posterior_reconstruction(
        self, inputs: InputBatch, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Reconstruct targets through the training-only expression posterior."""
        outputs = self.deterministic_prediction(inputs)
        context = outputs["context"]
        latent = self.posterior(target, context)
        prediction = self.prediction_from_latent(outputs["expression"], context, latent)
        return {"prediction": prediction, "latent": latent, **outputs}

    def controls(self, inputs: InputBatch, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return posterior, prior-center, and shuffled-latent diagnostics."""
        outputs = self.posterior_reconstruction(inputs, target)
        context = outputs["context"]
        latent = outputs["latent"]
        center, _ = self.prior_parameters(context)
        shuffled = latent[torch.randperm(len(latent), device=latent.device)]
        return {
            "posterior_reconstruction": outputs["prediction"],
            "zero_latent": self.prediction_from_latent(outputs["expression"], context, center),
            "shuffled_posterior": self.prediction_from_latent(
                outputs["expression"], context, shuffled
            ),
        }

    def training_loss(
        self, inputs: InputBatch, target: torch.Tensor, pcc_weight: float, config: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine reconstruction, deterministic, and prior-matching losses."""
        outputs = self.posterior_reconstruction(inputs, target)
        reconstruction, recon_metrics = point_loss(outputs["prediction"], target, pcc_weight)
        deterministic_loss, _ = point_loss(outputs["expression"], target, pcc_weight)
        mean, scale = self.prior_parameters(outputs["context"])
        # Compare posterior samples with the chosen prior in standardized space.
        standardized = (outputs["latent"] - mean) / scale.clamp_min(1e-6)
        reference = torch.randn_like(standardized)
        mmd = imq_mmd(standardized, reference)
        model = config["model"]
        total = (
            reconstruction
            + float(model.get("deterministic_weight", 1.0)) * deterministic_loss
            + float(model.get("mmd_weight", 0.1)) * mmd
        )
        return total, {
            **recon_metrics,
            "deterministic_loss": deterministic_loss,
            "mmd": mmd,
            "total": total,
        }

    def forward(self, inputs: InputBatch) -> torch.Tensor:
        """Return the default eight-draw predictive mean."""
        return self.sample(inputs, n_draws=8)["predictive_mean"]
