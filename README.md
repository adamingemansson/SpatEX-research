# SpatEX

SpatEX predicts spatial gene expression from cached UNI2 features and spot
coordinates. Measured expression is used as a target during training but is
never part of deployable model input.

Three configurations share the same deterministic backbone:

- `deterministic`: SpatEX point prediction;
- `wae_standard`: SpatEX with a WAE-MMD residual and `N(0, I)` prior;
- `wae_conditional`: SpatEX with a WAE-MMD residual and H&E-conditioned prior.

Data, UNI2 caches, fitted gene structure, checkpoints, and results stay outside
the repository and are supplied through configuration paths.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,test]'
```

## Prepare data

```bash
spatex-prepare \
  --source-manifest /external/dataset_manifest.json \
  --hest-root /external/hest1k \
  --uni2-cache /external/UNI2/cache/uni2_gen3_spot_cache \
  --output-root /external/spatex_prepared

spatex-fit-structure \
  --manifest /external/spatex_prepared/manifest.json \
  --output /external/spatex_prepared/centered_gene_structure.pt \
  --rank 64
```

## Train

```bash
spatex-launch \
  --config configs/deterministic.yaml \
  --config configs/wae_standard.yaml \
  --config configs/wae_conditional.yaml \
  --gpus 0,2,5 \
  --runs-root /external/SpatEX-runs \
  --name spatex_main \
  --tensorboard-port 55009
```

The launcher keeps resolved configs, logs, PIDs, checkpoints, TensorBoard data,
and latest-run pointers inside `SpatEX-runs`. It does not write to the data
directory. Smoke runs are created only when `--smoke-steps` is passed and are
kept under `SpatEX-runs/smoke/`.

## Evaluate and predict

```bash
spatex-evaluate \
  --config configs/deterministic.yaml \
  --checkpoint /external/run/checkpoints/best.pt \
  --output /external/evaluation.json

spatex-predict \
  --config configs/deterministic.yaml \
  --checkpoint /external/run/checkpoints/best.pt \
  --slide /external/slide_inputs.npz \
  --output /external/predictions.npz
```

Inference NPZ files contain `image_features`, normalized `coordinates`,
`image_available`, and `barcodes`. See `ARCHITECTURE.md` for model details and
WAE evaluation paths.

This repository is intended for fresh runs and does not load checkpoints from
the earlier experimental codebase.
