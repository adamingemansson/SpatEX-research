"""Train SpatEX or either WAE-MMD variant.

Training samples local spatial fields, while model selection uses complete
validation slides and the same primary prediction path used at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from spatex.checkpoint import load_checkpoint, save_checkpoint
from spatex.config import load_config
from spatex.data import PreparedManifest, SlideCache, SlideSpec
from spatex.losses import point_loss
from spatex.metrics import macro_gene_pcc, rmse
from spatex.models.deterministic import SpatEX
from spatex.models.factory import build_model
from spatex.models.legacy_exact import LegacyParallelGatedSpatEX
from spatex.models.wae import SpatEXWAE
from spatex.structure import load_structure
from spatex.tensorboard import (
    example_map,
    latent_pca_figure,
    spatial_gene_metrics,
    spatial_prediction_figure,
)


def _writer(path: Path):
    """Create a TensorBoard writer when the optional dependency is installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(path))
    except ImportError:
        return None


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    specs: tuple[SlideSpec, ...],
    n_genes: int,
    device: torch.device,
    wae_draws: int,
    *,
    pcc_weight: float = 0.1,
    writer=None,
    step: int | None = None,
    gene_names: tuple[str, ...] = (),
    logging: dict | None = None,
) -> dict[str, float]:
    """Average gene-wise spatial PCC and RMSE across validation slides."""
    model.eval()
    pcc_values: list[float] = []
    rmse_values: list[float] = []
    deterministic_pcc: list[float] = []
    deterministic_rmse: list[float] = []
    logging = {} if logging is None else logging
    image_every = int(logging.get("image_every", 0))
    latent_every = int(logging.get("latent_pca_every", 0))
    first_step = int(logging.get("first_diagnostic_step", image_every))
    write_images = (
        writer is not None
        and step is not None
        and image_every > 0
        and (step == first_step or step % image_every == 0)
    )
    write_latent = (
        isinstance(model, SpatEXWAE)
        and writer is not None
        and step is not None
        and latent_every > 0
        and (step == first_step or step % latent_every == 0)
    )
    write_gene_scalars = (
        writer is not None
        and step is not None
        and bool(logging.get("slide_gene_scalars", True))
    )
    examples = (
        example_map(logging, gene_names)
        if write_images or write_gene_scalars
        else {}
    )
    latent_values: list[torch.Tensor] = []
    latent_labels: list[str] = []
    validation_cache = SlideCache(n_genes, capacity=1)
    for spec in specs:
        record = validation_cache.get(spec)
        inputs, target = record.whole_slide(device)
        if isinstance(model, SpatEXWAE):
            paths = model.sample(inputs, wae_draws)
            prediction = paths["predictive_mean"]
            deterministic = paths["deterministic"]
            if write_latent:
                latent = model.posterior(target, paths["context"])
                per_slide = max(
                    1,
                    int(logging.get("latent_pca_max_points", 5000)) // len(specs),
                )
                indices = torch.linspace(
                    0, len(latent) - 1, min(per_slide, len(latent)), device=device
                ).long()
                latent_values.append(latent[indices].cpu())
                latent_labels.extend([record.organ] * len(indices))
        else:
            prediction = model(inputs)
            deterministic = prediction
        pcc_values.append(float(macro_gene_pcc(prediction, target).cpu()))
        rmse_values.append(float(rmse(prediction, target).cpu()))
        deterministic_pcc.append(float(macro_gene_pcc(deterministic, target).cpu()))
        deterministic_rmse.append(float(rmse(deterministic, target).cpu()))
        for gene_index in examples.get(record.sample_id, ()):
            gene = gene_names[gene_index]
            if write_gene_scalars:
                roles = {"primary": prediction}
                if isinstance(model, SpatEXWAE):
                    roles["deterministic"] = deterministic
                for role, role_prediction in roles.items():
                    metrics = spatial_gene_metrics(
                        target[:, gene_index],
                        role_prediction[:, gene_index],
                        pcc_weight=pcc_weight,
                    )
                    for name, value in metrics.items():
                        if math.isfinite(value):
                            writer.add_scalar(
                                f"slide_gene/{record.sample_id}/{gene}/{role}/{name}",
                                value,
                                step,
                            )
            if write_images:
                figure = spatial_prediction_figure(
                    inputs.coordinates,
                    target[:, gene_index],
                    prediction[:, gene_index],
                    sample_id=record.sample_id,
                    gene=gene,
                )
                writer.add_figure(
                    f"whole_slide/{record.sample_id}/{gene}",
                    figure,
                    step,
                    close=True,
                )
    if write_latent and latent_values:
        figure = latent_pca_figure(
            torch.cat(latent_values),
            latent_labels,
            max_points=int(logging.get("latent_pca_max_points", 5000)),
        )
        writer.add_figure("latent/posterior_z_pca_by_organ", figure, step, close=True)
    model.train()
    values = {
        "pcc": float(np.mean(pcc_values)),
        "rmse": float(np.mean(rmse_values)),
        "deterministic_pcc": float(np.mean(deterministic_pcc)),
        "deterministic_rmse": float(np.mean(deterministic_rmse)),
    }
    values["pcc_loss"] = 1.0 - values["pcc"]
    values["total"] = values["rmse"] + pcc_weight * values["pcc_loss"]
    values["deterministic_total"] = (
        values["deterministic_rmse"]
        + pcc_weight * (1.0 - values["deterministic_pcc"])
    )
    return values


def train(config_path: str | Path, resume: str | Path | None = None) -> Path:
    """Train one configured run until its step or wall-clock limit."""
    config = load_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cuda"))
    manifest = PreparedManifest.load(config["data"]["manifest"])
    structure = load_structure(config["data"]["structure"])
    if manifest.gene_names != structure.gene_names:
        raise ValueError("prepared manifest and structure have different gene order")
    model = build_model(config, structure).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start_step = 0
    if resume is not None:
        start_step = int(
            load_checkpoint(resume, model, config, manifest.gene_names, optimizer)["step"]
        )
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = (Path(config["training"]["output_root"]) / f"{config['run_name']}_{timestamp}")
    run_root = run_root.expanduser().resolve()
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    writer = _writer(run_root / "tensorboard")
    train_specs = manifest.for_split("train")
    validation_specs = manifest.for_split("validation")
    if not train_specs or not validation_specs:
        raise ValueError("training and validation slides are both required")
    generator = np.random.default_rng(seed + start_step)
    training = config["training"]
    slide_cache = SlideCache(
        len(manifest.gene_names), int(training.get("slide_cache_size", 2))
    )
    steps_per_slide = max(1, int(training.get("steps_per_slide", 20)))
    field_size = int(config["data"]["field_size"])
    total_steps = int(training["total_steps"])
    deadline = time.monotonic() + 3600.0 * float(training["max_hours"])
    validation_every = int(training["validation_every"])
    checkpoint_every = int(training["checkpoint_every"])
    logging = config.get("tensorboard", {})
    scalar_every = max(1, int(logging.get("scalar_every", 50)))
    best_validation = float("inf")
    final_step = start_step
    for step in range(start_step + 1, total_steps + 1):
        if time.monotonic() >= deadline:
            break
        if (step - start_step - 1) % steps_per_slide == 0:
            active_spec = train_specs[int(generator.integers(len(train_specs)))]
        record = slide_cache.get(active_spec)
        # Each step samples one local spatial field from one training slide.
        inputs, target = record.field(field_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, SpatEXWAE):
            loss, metrics = model.training_loss(
                inputs, target, float(training["pcc_weight"]), config
            )
        elif isinstance(model, (SpatEX, LegacyParallelGatedSpatEX)):
            prediction = model(inputs)
            loss, metrics = point_loss(
                prediction,
                target,
                float(training["pcc_weight"]),
                pcc_mode=str(training.get("pcc_mode", "gene_wise")),
            )
        else:
            raise TypeError(type(model))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        clip_norm = float(training.get("gradient_clip_norm", 5.0))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        final_step = step
        if writer is not None and (step == 1 or step % scalar_every == 0):
            for name, value in metrics.items():
                writer.add_scalar(f"train/{name}", float(value.detach().cpu()), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
        if step == 1 or step % 50 == 0:
            values = ", ".join(
                f"{name}={float(value.detach().cpu()):.6f}"
                for name, value in metrics.items()
            )
            print(f"[step {step}] train: {values}", flush=True)
        validation: dict[str, float] = {}
        if step % validation_every == 0:
            # Model selection uses the same complete-slide path used at inference.
            validation = validate(
                model,
                validation_specs,
                len(manifest.gene_names),
                device,
                int(config["evaluation"]["wae_draws"]),
                pcc_weight=float(training["pcc_weight"]),
                writer=writer,
                step=step,
                gene_names=manifest.gene_names,
                logging=logging,
            )
            print(f"[step {step}] validation: {validation}", flush=True)
            if writer is not None:
                for name, value in validation.items():
                    writer.add_scalar(f"validation/{name}", value, step)
            if validation["total"] < best_validation:
                best_validation = validation["total"]
                save_checkpoint(
                    run_root / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    config,
                    manifest.gene_names,
                    step,
                    validation,
                )
        if step % checkpoint_every == 0:
            save_checkpoint(
                run_root / "checkpoints" / "last.pt",
                model,
                optimizer,
                config,
                manifest.gene_names,
                step,
                validation,
            )
    save_checkpoint(
        run_root / "checkpoints" / "last.pt",
        model,
        optimizer,
        config,
        manifest.gene_names,
        final_step,
        {},
    )
    if writer is not None:
        writer.close()
    print(f"training finished at step {final_step}: {run_root}", flush=True)
    return run_root


def main() -> None:
    """Start training from the command line."""
    parser = argparse.ArgumentParser(description="Train deterministic or WAE-MMD SpatEX model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
